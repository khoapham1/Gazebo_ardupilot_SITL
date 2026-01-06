import os
import time
import math
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor

import cv2
import requests

from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
from ultralytics import YOLO


# =========================
# YOLO CONFIG
# =========================
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")  # can be a local .pt path
YOLO_IMG_SIZE   = int(os.environ.get("YOLO_IMG_SIZE", "640"))
YOLO_CONF       = float(os.environ.get("YOLO_CONF", "0.35"))
YOLO_IOU        = float(os.environ.get("YOLO_IOU", "0.45"))
YOLO_CLASSES    = [0]  # COCO: person

# How often we run inference / send updates
PROCESS_INTERVAL_SEC = float(os.environ.get("PROCESS_INTERVAL_SEC", "0.15"))  # ~6-7 fps inference throttle
SEND_INTERVAL_SEC    = float(os.environ.get("SEND_INTERVAL_SEC", "2.0"))      # send GPS ping at most every N seconds
MIN_MOVE_TO_NEW_PING_M = float(os.environ.get("MIN_MOVE_TO_NEW_PING_M", "2.0"))

bridge = CvBridge()

_latest_frame_lock = threading.Lock()
_latest_frame_jpeg = None
_image_streamer_node = None
_image_streamer_executor = None
_image_streamer_thread = None

class ImageStreamerNode(Node):
    """
    ROS 2 node that subscribes to processed topic and updates latest JPEG for MJPEG streaming.
    """
    def __init__(self, topic_name='/UAV/forward/image_new', queue_size=2):
        super().__init__('image_streamer_node')
        self.topic_name = topic_name
        self.sub = self.create_subscription(Image, self.topic_name, self.cb_image, queue_size)
        self.bridge = CvBridge()
        self.get_logger().info(f"ImageStreamerNode subscribing to {self.topic_name}")

    def cb_image(self, msg):
        global _latest_frame_jpeg, _latest_frame_lock
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ret, jpeg = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                with _latest_frame_lock:
                    _latest_frame_jpeg = jpeg.tobytes()
        except Exception as e:
            self.get_logger().error(f"Error in cb_image: {e}")

def start_image_streamer(topic_name='/UAV/forward/image_new'):
    """
    Start background ROS2 node that listens to processed camera topic and updates latest JPEG.
    Safe to call multiple times.
    """
    global _image_streamer_node, _image_streamer_executor, _image_streamer_thread
    if _image_streamer_node is not None:
        return
    if not rclpy.ok():
        try:
            rclpy.init(args=None)
        except Exception:
            pass
    _image_streamer_node = ImageStreamerNode(topic_name=topic_name)
    _image_streamer_executor = SingleThreadedExecutor()
    _image_streamer_executor.add_node(_image_streamer_node)

    def spin_loop():
        try:
            while rclpy.ok():
                _image_streamer_executor.spin_once(timeout_sec=0.1)
                time.sleep(0.01)
        except Exception:
            pass
        finally:
            try:
                _image_streamer_executor.remove_node(_image_streamer_node)
                _image_streamer_node.destroy_node()
            except Exception:
                pass
    _image_streamer_thread = threading.Thread(target=spin_loop, daemon=True)
    _image_streamer_thread.start()

def stop_image_streamer():
    global _image_streamer_node, _image_streamer_executor, _image_streamer_thread
    if _image_streamer_node is None:
        return
    try:
        _image_streamer_executor.shutdown()
    except Exception:
        pass
    _image_streamer_node = None
    _image_streamer_executor = None
    _image_streamer_thread = None

def get_lastest_frame():
    """Return latest JPEG bytes or None."""
    global _latest_frame_jpeg, _latest_frame_lock
    with _latest_frame_lock:
        return _latest_frame_jpeg

def _approx_distance_m(lat1, lon1, lat2, lon2):
    # same approximation used in your server telemetry_loop
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return math.sqrt((dlat**2) + (dlon**2)) * 1.113195e5

class DroneController:
    def __init__(self, connection_str='tcp:127.0.0.1:5763', takeoff_height=3):
        self.connection_str = connection_str
        print("Connecting to vehicle on", connection_str)
        self.vehicle = connect(connection_str, wait_ready=True, timeout=120)
        self.takeoff_height = takeoff_height

        if not rclpy.ok():
            rclpy.init(args=None)

        self.flown_path = []
        self.ros_node = None
        self.executor = None
        self.detect_thread = None
        self.detect_running = False

    # stream image for web UI
    def start_image_stream(self, topic_name='/UAV/forward/image_new'):
        try:
            start_image_streamer(topic_name=topic_name)
            print("Started image streamer on topic", topic_name)
        except Exception as e:
            print("Failed to start image streamer:", e)

    def stop_image_stream(self):
        try:
            stop_image_streamer()
            print("Stopped image streamer")
        except Exception as e:
            print("Failed to stop image streamer:", e)

    # send detections to server (realtime ping on map)
    def send_people_detections_to_server(self, detections: dict):
        try:
            response = requests.post(
                'http://localhost:5000/update_people_detections',
                json={'detections': detections},
                timeout=2
            )
            if response.status_code != 200:
                print(f"Failed to send detections, status code: {response.status_code}")
        except Exception as e:
            print(f"Error sending detections to server: {e}")

    # MAVLink helpers (kept)
    def send_local_ned_velocity(self, vx, vy, vz):
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def set_speed(self, speed):
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            1,
            speed,
            -1, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        print(f"Set speed to {speed} m/s")

    def get_distance_meters(self, targetLocation, currentLocation):
        dLat = targetLocation.lat - currentLocation.lat
        dLon = targetLocation.lon - currentLocation.lon
        return math.sqrt((dLon * dLon) + (dLat * dLat)) * 1.113195e5

    def goto(self, targetLocation, tolerance=2.0, timeout=60, speed=2.5):
        distanceToTargetLocation = self.get_distance_meters(targetLocation, self.vehicle.location.global_relative_frame)
        self.set_speed(speed)
        self.vehicle.simple_goto(targetLocation, groundspeed=speed)
        start_dist = distanceToTargetLocation
        start_time = time.time()
        while self.vehicle.mode.name == "GUIDED" and time.time() - start_time < timeout:
            currentDistance = self.get_distance_meters(targetLocation, self.vehicle.location.global_relative_frame)
            current_pos = self.vehicle.location.global_relative_frame
            if current_pos.lat and current_pos.lon:
                self.flown_path.append([current_pos.lat, current_pos.lon])
            if currentDistance < max(tolerance, start_dist * 0.01):
                print("Reached target waypoint")
                return True
            time.sleep(0.02)
        print("Timeout reaching waypoint, proceeding anyway")
        return False

    def arm_and_takeoff(self, targetHeight):
        while not self.vehicle.is_armable:
            print('Waiting for vehicle to become armable')
            time.sleep(1)
        self.vehicle.mode = VehicleMode('GUIDED')
        while self.vehicle.mode != 'GUIDED':
            print('Waiting for GUIDED...')
            time.sleep(1)
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print('Arming...')
            time.sleep(1)
        self.vehicle.simple_takeoff(targetHeight)
        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            print('Altitude: %.2f' % (alt if alt else 0.0))
            if alt and alt >= 0.95 * targetHeight:
                break
            time.sleep(1)
        print("Reached takeoff altitude")

    # detection thread
    def start_detection(self):
        if self.detect_running:
            print("Detection already running")
            return
        self.ros_node = DroneNode(self)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ros_node)
        self.detect_running = True
        self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self.detect_thread.start()
        print("Started YOLO detection thread")

    def _detect_loop(self):
        while rclpy.ok() and self.detect_running and self.vehicle.armed:
            self.executor.spin_once(timeout_sec=0.1)

    def stop_detection(self):
        if not self.detect_running:
            return
        self.detect_running = False
        if self.detect_thread:
            self.detect_thread.join(timeout=2.0)
        if self.executor and self.ros_node:
            try:
                self.executor.remove_node(self.ros_node)
            except Exception:
                pass
            try:
                self.ros_node.destroy_node()
            except Exception:
                pass
        self.ros_node = None
        self.executor = None
        print("Stopped YOLO detection thread")

    def fly_and_detect_people_with_waypoints(self, waypoints, loiter_alt=3):
        """
        Fly GPS waypoints + run YOLOv8n person detection in background,
        and ping detections (current UAV GPS) to web map in realtime.
        """
        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        self.flown_path = []

        print("Arming and taking off")
        self.arm_and_takeoff(loiter_alt)
        time.sleep(1)

        # Start detection while flying (replaces ArUco scanning)
        self.start_detection()

        # Store home
        home_lat = self.vehicle.location.global_relative_frame.lat
        home_lon = self.vehicle.location.global_relative_frame.lon
        wp_home = LocationGlobalRelative(home_lat, home_lon, loiter_alt)
        print(f"Home recorded at lat={home_lat:.6f}, lon={home_lon:.6f}")

        # Fly through waypoints
        for i, wp in enumerate(waypoints[1:]):
            wp_loc = LocationGlobalRelative(wp[0], wp[1], loiter_alt)
            print(f"Flying to waypoint {i+1}: {wp[0]}, {wp[1]}")
            self.goto(wp_loc)

        # Return home
        print("Returning to home")
        self.goto(wp_home)

        # Stop detection before landing
        self.stop_detection()

        # Land at home (normal LAND)
        print("Landing at home")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode != "LAND":
            print("Waiting for LAND mode...")
            time.sleep(1)

        while self.vehicle.armed:
            print("Waiting for disarming...")
            time.sleep(1)

        print("Mission complete")

class DroneNode(Node):
    def __init__(self, controller):
        node_name = f"drone_node_for_yolo_{int(time.time()*1000)}"
        super().__init__(node_name)
        self.controller = controller
        self.vehicle = controller.vehicle

        self.newimg_pub = self.create_publisher(Image, '/UAV/forward/image_new', 10)
        self.subscription = self.create_subscription(Image, '/UAV/forward/image_raw', self.msg_receiver, 10)

        # YOLO model
        self.model = None
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.get_logger().info(f"Loaded YOLO model: {YOLO_MODEL_PATH}")
        except Exception as e:
            self.get_logger().error(f"Failed to load ultralytics/YOLO: {e}. Video will stream without detections.")

        self.imgsz = YOLO_IMG_SIZE
        self.conf = YOLO_CONF
        self.iou = YOLO_IOU
        self.classes = YOLO_CLASSES

        self.last_process_time = 0.0
        self.last_send_time = 0.0
        self.last_sent_lat = None
        self.last_sent_lon = None
        self.seq = 0
        self.detections = {}

    def _should_send(self, now, lat, lon):
        if now - self.last_send_time < SEND_INTERVAL_SEC:
            return False
        if self.last_sent_lat is None or self.last_sent_lon is None:
            return True
        try:
            d = _approx_distance_m(self.last_sent_lat, self.last_sent_lon, lat, lon)
            return d >= MIN_MOVE_TO_NEW_PING_M
        except Exception:
            return True

    def msg_receiver(self, message):
        now = time.time()
        if now - self.last_process_time < PROCESS_INTERVAL_SEC:
            return
        self.last_process_time = now

        try:
            cv_image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        if self.model is None:
            try:
                new_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                self.newimg_pub.publish(new_msg)
            except Exception:
                pass
            return

        found_person = False
        best_conf = 0.0

        try:
            results = self.model.predict(
                source=cv_image,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                verbose=False
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                r0 = results[0]
                names = getattr(r0, "names", None)

                for b in r0.boxes:
                    cls = int(b.cls[0]) if hasattr(b, "cls") else -1
                    conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = "person" if (names is None or cls not in names) else str(names[cls])
                    cv2.putText(cv_image, f"{label} {conf:.2f}", (x1, max(0, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    if label == "person" or cls == 0:
                        found_person = True
                        best_conf = max(best_conf, conf)

        except Exception as e:
            self.get_logger().error(f"YOLO inference error: {e}")

        if found_person:
            lat = self.vehicle.location.global_relative_frame.lat
            lon = self.vehicle.location.global_relative_frame.lon
            if lat is not None and lon is not None and self._should_send(now, lat, lon):
                self.seq += 1
                det_id = f"det_{int(now*1000)}_{self.seq:03d}"
                det = {"lat": float(lat), "lon": float(lon), "conf": float(best_conf), "ts": float(now)}

                self.detections[det_id] = det
                self.last_send_time = now
                self.last_sent_lat = float(lat)
                self.last_sent_lon = float(lon)

                print(f"[DETECT] person conf={best_conf:.2f} -> lat={lat:.6f}, lon={lon:.6f} (id={det_id})")
                try:
                    self.controller.send_people_detections_to_server({det_id: det})
                except Exception as e:
                    print("Error sending detection:", e)

        try:
            new_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.newimg_pub.publish(new_msg)
        except Exception:
            pass

_controller = None

def get_controller(connection_str='tcp:127.0.0.1:5763', takeoff_height=3):
    global _controller
    if _controller is None:
        _controller = DroneController(connection_str=connection_str, takeoff_height=takeoff_height)
    return _controller