
#!/usr/bin/env python3
# drone_control.py (record aFnd interpolate flown path for exact return trajectory
import time
import math
import threading
from matplotlib import markers
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
from rclpy.executors import SingleThreadedExecutor
import requests
import os

# YOLOv5
try:
    import torch
except Exception:
    torch = None

# ---- camera / aruco params (copy from yours) ---
aruco = cv2.aruco
bridge = CvBridge()
ids_to_find = [1, 2]


# find_aruco giờ là động, sẽ set qua class
marker_sizes = [60, 10]          # cm
marker_heights = [10, 3]         # m, altitude thresholds


horizontal_res = 1280
vertical_res = 720
horizontal_fov = 62.2 * (math.pi / 180)
vertical_fov = 48.8 * (math.pi / 180)

dist_coeff = [0.0, 0.0, 0.0, 0.0, 0.0]
camera_matrix = [[530.8269276712998, 0.0, 320.5],
                 [0.0, 530.8269276712998, 240.5],
                 [0.0, 0.0, 1.0]]

np_camera_matrix = np.array(camera_matrix)
np_dist_coeff = np.array(dist_coeff)

time_to_wait = 0.1
time_last = 0

_latest_frame_lock = threading.Lock()
_latest_frame_jpeg = None
_image_streamer_node = None
_image_streamer_executor = None
_image_streamer_thread = None

class ImageStreamerNode(Node):
    """
    ROS 2 node that subscribes to '/UAV/forward/image_new'
    """
    def __init__(self, topic_name='/UAV/forward/image_new', queue_size=2):
        super().__init__('image_streamer_node')
        self.topic_name = topic_name
        self.sub = self.create_subscription(Image, self.topic_name, self.cb_image, queue_size)  # Fix: cd_image -> cb_image
        self.bridge = CvBridge()
        self.get_logger().info(f"ImageStreamerNode subscribing to {self.topic_name}")

    def cb_image(self, msg):
        global _latest_frame_jpeg, _latest_frame_lock
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # Fix: brg8 -> bgr8
            ret, jpeg = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # Fix: imenode -> imencode
            if ret:
                with _latest_frame_lock:
                    _latest_frame_jpeg = jpeg.tobytes()
        except Exception as e:
            self.get_logger().error(f"Error in cb_image: {e}")

# ---- Streaming Image ----
def start_image_streamer(topic_name='/UAV/forward/image_new'):
    """
    Start background ROS2 node that listens to camera topic and updates latest JPEG.
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

# ---- Stop stream image ----
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
    """
    Return lastest JPEG bytes or None.
    """
    global _latest_frame_jpeg, _latest_frame_lock  # Fix: _latest_frame_jpeg -> _latest_frame_lock
    with _latest_frame_lock:
        return _latest_frame_jpeg

# ---- DroneController class ----

class DroneController:
    def __init__(self, connection_str='tcp:127.0.0.1:5763', takeoff_height=3):
        " connection_str = 'udp:0.0.0.0:14450'"
        """
        Create DroneController and connect to vehicle.
        """
        self.connection_str = connection_str
        print("Connecting to vehicle on", connection_str)
        self.vehicle = connect(connection_str, wait_ready=True, timeout=120)
        #self.vehicle = connect('udp:0.0.0.0:14550')
        
        # set some landing params
        try:
            self.vehicle.parameters['PLND_ENABLED'] = 1
            self.vehicle.parameters['PLND_TYPE'] = 1  # ArUco-based precision landing
            self.vehicle.parameters['PLND_EST_TYPE'] = 0
            self.vehicle.parameters['LAND_SPEED'] = 20

        except Exception:
            print("Failed to set some landing parameters")
        self.takeoff_height = takeoff_height
        if not rclpy.ok():
            rclpy.init(args=None)
        self.flown_path = []  # Store actual flown path
        self.ros_node = None
        self.executor = None
        self.scan_thread = None
        self.scan_running = False
        # Danh sách ArUco IDs cần detect (mặc định)
        self.find_aruco = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        # ===== Pause-on-detection (hold position) =====
        # When detecting "person_in_water" (or other configured labels), pause (hold) for N seconds then continue.
        self._pause_lock = threading.Lock()
        self._pause_until = 0.0
        self._pause_reason = None
        self._pause_hold_loc = None
        self._pause_last_trigger = 0.0
        self._pause_hold_s = float(os.getenv("PERSON_HOLD_SECONDS", "10"))  # seconds to hold
        self._pause_cooldown_s = float(os.getenv("PERSON_HOLD_COOLDOWN_SECONDS", "15"))  # prevent re-trigger spam
        labels_env = os.getenv("PERSON_HOLD_LABELS", "person_in_water,drowning")
        self._pause_labels = set([s.strip().lower() for s in labels_env.split(",") if s.strip()])
    
    # start stream image
    def start_image_stream(self, topic_name='/UAV/forward/image_new'):
        try:
            start_image_streamer(topic_name=topic_name)
            print("Started image streamer on topic", topic_name)
        except Exception as e:
            print("Failed to start image streamer:", e)
    def stop_image_streamer(self):
        try:
            stop_image_streamer()
            print("Stopped image streamer")
        except Exception as e:
            print("Failed to stop image streamer:", e)
    
    def set_find_aruco(self, ids):
        if isinstance(ids, list) and all(isinstance(id, int) for id in ids):
            self.find_aruco = ids
        else:
            raise ValueError("Invalid ArUco IDs. Must be list of integers.")

    def send_aruco_marker_to_server(self, markers):

        """Send detected person-in-water zones to server (keeps old method name for compatibility)."""
        try:
            response = requests.post(
                'http://localhost:5000/update_person_zones',
                json={'zones': markers},
                timeout=2
            )
            if response.status_code == 200:
                print("Successfully sent person zones to server")
            else:
                print(f"Failed to send person zones, status code: {response.status_code}")
        except Exception as e:
            print(f"Error sending person zones to server: {e}")

        # ===== Pause helpers =====
    def request_pause(self, reason="person_in_water", hold_s=None):
        """Request UAV to hold position for hold_s seconds (non-blocking).
        Returns True if pause activated/extended, False if ignored (cooldown/label).
        """
        r = (reason or "").strip().lower()
        if r not in getattr(self, "_pause_labels", set()):
            return False
        try:
            hold_s = float(hold_s) if hold_s is not None else float(self._pause_hold_s)
        except Exception:
            hold_s = 5.0

        now = time.time()
        with self._pause_lock:
            # If already pausing, ignore new triggers so the hold doesn't get extended forever
            if now < self._pause_until:
                return False

            # cooldown to avoid re-trigger every frame
            if (now - self._pause_last_trigger) < float(self._pause_cooldown_s):
                return False

            self._pause_last_trigger = now
            self._pause_until = now + hold_s
            self._pause_reason = r

            # snapshot hold location (best-effort)
            try:
                loc = self.vehicle.location.global_relative_frame
                if loc.lat is not None and loc.lon is not None:
                    alt = loc.alt if loc.alt is not None else self.takeoff_height
                    self._pause_hold_loc = (float(loc.lat), float(loc.lon), float(alt))
                else:
                    self._pause_hold_loc = None
            except Exception:
                self._pause_hold_loc = None

        print(f"[PAUSE] Hold requested: reason={r}, hold_s={hold_s}")
        # NOTE: removed invalid cv2.putText call (no image buffer in request_pause)
        return True

    def _is_pausing(self):
        with self._pause_lock:
            return time.time() < float(self._pause_until)

    def _pause_remaining(self):
        with self._pause_lock:
            return max(0.0, float(self._pause_until) - time.time())

    def _hold_position_once(self):
        """Send a one-shot position-hold command (best-effort)."""
        try:
            with self._pause_lock:
                hl = self._pause_hold_loc
            if not hl:
                return
            lat, lon, alt = hl
            hold_loc = LocationGlobalRelative(lat, lon, alt)
            # very small speed so it doesn't drift away
            self.vehicle.simple_goto(hold_loc, groundspeed=0.1)
        except Exception:
            pass

# MAVLink helpers
    def send_local_ned_velocity(self, vx, vy, vz):
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            self.vehicle._master.target_system,
            self.vehicle._master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            1479,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0.0,      # yaw (ignored)
            0  # yaw_rate USED (0.0 => giữ yaw)
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def send_land_message(self, x, y):
        msg = self.vehicle.message_factory.landing_target_encode(
            0,
            0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            x,
            y,
            0,
            0,
            0)
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

    # Core flight primitives
    def get_distance_meters(self, targetLocation, currentLocation):
        dLat = targetLocation.lat - currentLocation.lat
        dLon = targetLocation.lon - currentLocation.lon
        return math.sqrt((dLon * dLon) + (dLat * dLat)) * 1.113195e5

    def goto(self, targetLocation, tolerance=2.0, timeout=60, speed=0.7):
        """Go to a waypoint with pause-aware timeout."""
        if speed < 0.1 or speed > 5.0:
            print(f"Tốc độ {speed} m/s không hợp lệ, đặt về 0.7 m/s")
            speed = 0.7
        if not self.vehicle:
            return False
        

        distanceToTargetLocation = self.get_distance_meters(
            targetLocation, self.vehicle.location.global_relative_frame
        )
        self.set_speed(speed)
        self.vehicle.simple_goto(targetLocation, groundspeed=speed)

        start_dist = distanceToTargetLocation
        start_time = time.time()

        # pause-aware timeout
        pause_accum = 0.0
        pause_start = None
        hold_sent = False

        while self.vehicle.mode.name == "GUIDED" and self.vehicle.armed:
            now = time.time()

            # effective elapsed (ignore time spent pausing)
            elapsed = now - start_time - pause_accum
            if pause_start is not None:
                elapsed -= (now - pause_start)
            if elapsed > timeout:
                break

            # ===== pause handling =====
            if self._is_pausing():
                if pause_start is None:
                    pause_start = now
                    hold_sent = False
                    print(f"[PAUSE] Holding position for {self._pause_remaining():.1f}s (reason={self._pause_reason})")
                if not hold_sent:
                    self._hold_position_once()
                    hold_sent = True
                try:
                    # keep publishing zero-velocity to counter simple_goto motion
                    self.send_local_ned_velocity(0, 0, 0)
                except Exception:
                    pass
                time.sleep(0.1)
                continue
            else:
                if pause_start is not None:
                    pause_accum += (now - pause_start)
                    pause_start = None
                    hold_sent = False
                    print("[PAUSE] Resume mission")
                    try:
                        # Re-issue the original goto command so we actually continue to the waypoint
                        self.set_speed(speed)
                        self.vehicle.simple_goto(targetLocation, groundspeed=speed)
                    except Exception:
                        pass


            # normal navigation
            currentDistance = self.get_distance_meters(
                targetLocation, self.vehicle.location.global_relative_frame
            )

            # Record current position
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
            if alt >= 0.95 * targetHeight:
                break
            time.sleep(1)
        print("Reached takeoff altitude")
        return None
    
    def land_and_wait(self, timeout=120):
        """Land and wait until disarmed (or timeout)."""
        try:
            self.vehicle.mode = VehicleMode("LAND")
        except Exception as e:
            print("Failed to set LAND mode:", e)
            return False

        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if not self.vehicle.armed:
                    print("Landed (disarmed)")
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        print("land_and_wait timeout")
        return False


    def start_scanning(self):
        """
        Start a separate thread for scanning ArUco markers 3-17 during flight.
        """
        if self.scan_running:
            print("Scanning already running")
            return
        self.ros_node = DroneNode(self, mode='scan')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ros_node)
        self.scan_running = True
        self.scan_thread = threading.Thread(target=self._scan_loop)
        self.scan_thread.start()
        print("Started scanning for YOLOv5 person_in_water")

    def _scan_loop(self):
        while rclpy.ok() and self.scan_running and self.vehicle.armed:
            self.executor.spin_once(timeout_sec=0.1)

    def stop_scanning(self):
        """
        Stop the scanning thread and clean up.
        """
        if not self.scan_running:
            return
        self.scan_running = False
        if self.scan_thread:
            self.scan_thread.join()
        if self.executor and self.ros_node:
            self.executor.remove_node(self.ros_node)
            self.ros_node.destroy_node()
        self.ros_node = None
        self.executor = None
        print("Stopped scanning")

    def _start_aruco_processing(self, duration=30, mode='land'):
        """
        Start ROS2 node that subscribes to camera image and processes ArUco for specified mode while spinning for 'duration'
        """
        node = DroneNode(self, mode=mode)  # Pass the vehicle to DroneNode
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        start_time = time.time()
        try:
            while rclpy.ok() and time.time() - start_time < duration and self.vehicle.armed:
                executor.spin_once(timeout_sec=0.1)
                if mode == 'land':
                    # Check altitude to switch to LAND mode
                    alt = self.vehicle.location.global_relative_frame.alt
                    if alt is not None and alt < 1.0:  # PLND_ALT_LOW
                        print("Low altitude reached, switching to LAND mode")
                        self.vehicle.mode = VehicleMode("LAND")
                        while self.vehicle.mode != "LAND":
                            print("Waiting for LAND mode...")
                            time.sleep(1)
                        break
        finally:
            executor.remove_node(node)
            node.destroy_node()

    def interpolate_path(self, path, num_points=20):
        """
        Interpolate the recorded path to generate a smooth set of waypoints.
        """
        if not path or len(path) < 2:
            return path
        path = np.array(path)
        t = np.linspace(0, 1, len(path))
        t_new = np.linspace(0, 1, num_points)
        lat = np.interp(t_new, t, path[:, 0])
        lon = np.interp(t_new, t, path[:, 1])
        return [[lat[i], lon[i]] for i in range(num_points)]
    
    def fly_and_precision_land_with_waypoints(self, waypoints, loiter_alt=3, aruco_duration=30):
        """
        Fly through waypoints and LAND at the final waypoint.
        (Previously the mission thread ended before LAND completed, so the drone stayed flying.)
        """
        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        # Clear previous flown path
        self.flown_path = []

        # Takeoff
        print("Arming and taking off")
        self.arm_and_takeoff(loiter_alt)
        time.sleep(1)

        # Start YOLO scanning during flight
        self.start_scanning()

        # Fly intermediate waypoints (skip home at index 0, land at last)
        for i, wp in enumerate(waypoints[1:], start=1):
            wp_loc = LocationGlobalRelative(wp[0], wp[1], loiter_alt)
            print(f"Flying to waypoint {i}: {wp[0]}, {wp[1]}")
            self.goto(wp_loc)

        # Stop scanning before landing
        self.stop_scanning()

        print("Starting landing phase...")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode.name != "LAND":
            print("Waiting for LAND mode...")
            time.sleep(1)

        while self.vehicle.armed:
            print("Waiting for disarming...")
            time.sleep(1)

        print("Mission complete")
        return True


class DroneNode(Node):
    def __init__(self, controller, mode='scan'):
        node_name = f"drone_node_for_yolo{int(time.time()*1000)}"
        super().__init__(node_name)
        self.controller = controller
        self.vehicle = controller.vehicle
        self.mode = mode

        # Publish annotated image for web MJPEG stream
        self.newimg_pub = self.create_publisher(Image, '/UAV/forward/image_new', 10)
        # Subscribe to raw camera
        self.subscription = self.create_subscription(Image, '/UAV/forward/image_raw', self.msg_receiver, 10)

        # Send interval gui len server 2s
        self.last_send_time = 0.0
        self.send_interval = float(os.getenv("PERSON_SEND_INTERVAL", "2.0"))

        # Keep old attribute name to avoid changing controller logic
        # Format: { "<label>": {lat, lon, radius_m, conf, ts} }
        self.detected_marker = {}

        names_env = os.getenv("YOLOV5_TARGET_CLASS_NAMES", "person_in_water,swimming,drowning")
        self.target_labels = [s.strip().lower() for s in names_env.split(",") if s.strip()]

        # Confidence threshold
        self.conf_thres = float(os.getenv("YOLOV5_CONF", "0.35"))
        self.conf_thres_water = float(os.getenv("YOLOV5_CONF_WATER", "0.20"))  # person-in-water is usually harder

        # Default zone radius on map (meters)
        self.zone_radius_m = float(os.getenv("PERSON_ZONE_RADIUS_M", "5"))
        self.scale_zone_with_alt = os.getenv("ZONE_SCALE_WITH_ALT", "0").strip() not in ("0", "false", "False")

        # Water-mask heuristic (to turn "person" -> "person_in_water")
        self.enable_water_heuristic = os.getenv("WATER_HEURISTIC", "1").strip() not in ("0", "false", "False")
        # HSV ranges for water (tunable). Defaults match "blue ocean" sim reasonably.
        self.water_hsv_low = tuple(int(x) for x in os.getenv("WATER_HSV_LOW", "80,40,40").split(","))
        self.water_hsv_high = tuple(int(x) for x in os.getenv("WATER_HSV_HIGH", "140,255,255").split(","))
        self.water_ratio_thres = float(os.getenv("WATER_RATIO_THRES", "0.12"))  # % of water pixels (lower half of bbox)

        # Temporal confirmation to reduce false positives
        self.confirm_frames_person_in_water = int(os.getenv("CONFIRM_FRAMES_PERSON_IN_WATER", "2"))
        self.confirm_frames_drowning = int(os.getenv("CONFIRM_FRAMES_DROWNING", "3"))
        self.track_timeout_s = float(os.getenv("TRACK_TIMEOUT_S", "1.0"))

        # Simple per-label tracking
        self._tracks = {}  # label -> dict(count, last_ts, last_bbox, ema_conf)
        # Image center
        self.last_frame_center = None
        self.last_bbox_center = {}
        self.last_center_offset = {}

        # ===== Camera geometry for angle calculation (radians) =====
        # Defaults match your camera: 1280x720, HFOV=62.2deg, VFOV=48.8deg
        self.horizontal_res = int(os.getenv("HORIZONTAL_RES", "1280"))
        self.vertical_res   = int(os.getenv("VERTICAL_RES", "720"))
        self.horizontal_fov = float(os.getenv("HORIZONTAL_FOV_DEG", "62.2")) * (math.pi / 180.0)
        self.vertical_fov   = float(os.getenv("VERTICAL_FOV_DEG", "48.8")) * (math.pi / 180.0)

        # Store latest angles (label -> (x_ang, y_ang) in radians)
        self.last_bbox_angles = {}
        # Load YOLOv5
        self.yolo = self._load_yolov5()
        self.get_logger().info("DroneNode YOLOv5 ready")

    def _load_yolov5(self):
        if torch is None:
            raise RuntimeError("PyTorch (torch) is not installed in this environment.")

        repo = os.getenv("YOLOV5_REPO", os.path.expanduser("~/ai/yolov5"))
        weights = os.getenv("YOLOV5_WEIGHTS", "yolov5s.pt")  # set to your custom best.pt for person_in_water

        try:
            if os.path.isdir(repo):
                model = torch.hub.load(repo, 'custom', path=weights, source='local', force_reload=False)
            else:
                # Fallback: requires internet on first run
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5 model. repo={repo}, weights={weights}, err={e}")

        try:
            model.conf = self.conf_thres
        except Exception:
            pass
        return model

    
    def _preprocess(self, bgr):
        """Lightweight contrast boost for water scenes."""
        try:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge([l2, a, b])
            out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

            # mild gamma to lift dark water textures
            gamma = float(os.getenv("WATER_GAMMA", "1.1"))
            if abs(gamma - 1.0) > 1e-3:
                table = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255.0).astype(np.uint8)
                out = cv2.LUT(out, table)

            # optional mild sharpening (helps tiny heads in sim)
            if os.getenv("WATER_SHARPEN", "1").strip() not in ("0", "false", "False"):
                blur = cv2.GaussianBlur(out, (0, 0), 1.0)
                out = cv2.addWeighted(out, 1.25, blur, -0.25, 0)

            return out
        except Exception:
            return bgr

    def _water_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        low = np.array(self.water_hsv_low, dtype=np.uint8)
        high = np.array(self.water_hsv_high, dtype=np.uint8)
        mask = cv2.inRange(hsv, low, high)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        return mask

    @staticmethod
    def _clamp_bbox(x1, y1, x2, y2, w, h):
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2

    def _bbox_water_ratio(self, water_mask, x1, y1, x2, y2):
        """Return a water-score in [0..1] for bbox.
        Robust heuristic for aerial/sim scenes:
        - water ratio in lower part of bbox
        - water ratio in bottom strip
        - water ratio in expanded bbox (context water around the person)
        - bottom-center pixel water hit
        """
        try:
            h, w = water_mask.shape[:2]
            x1, y1, x2, y2 = self._clamp_bbox(x1, y1, x2, y2, w, h)

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # lower part (slightly more than half)
            y_mid = int(y1 + 0.45 * bh)
            roi_lower = water_mask[y_mid:y2, x1:x2]
            s_lower = float(np.mean(roi_lower) / 255.0) if roi_lower.size else 0.0

            # bottom strip (strong cue)
            y_bot = int(y1 + 0.80 * bh)
            roi_bottom = water_mask[y_bot:y2, x1:x2]
            s_bottom = float(np.mean(roi_bottom) / 255.0) if roi_bottom.size else 0.0

            # expanded bbox (context around person)
            pad = int(0.15 * max(bw, bh))
            x1e = max(0, x1 - pad)
            x2e = min(w - 1, x2 + pad)
            y1e = max(0, y1 - pad)
            y2e = min(h - 1, y2 + pad)
            roi_expand = water_mask[y_mid:y2e, x1e:x2e]
            s_expand = float(np.mean(roi_expand) / 255.0) if roi_expand.size else 0.0

            # bottom-center pixel hit
            cx = int((x1 + x2) * 0.5)
            cy = min(h - 1, y2 - 1)
            s_hit = 1.0 if water_mask[cy, cx] > 0 else 0.0

            # weighted max
            return float(max(s_lower, s_bottom * 1.1, s_expand * 0.9, s_hit))
        except Exception:
            return 0.0

    def _map_raw_to_label(self, raw_name, conf, water_ratio):
        """Map YOLO class name -> label we publish."""
        n = (raw_name or "").strip().lower()

        # custom weights might use these names
        if "drown" in n:
            return "drowning"
        if "swim" in n:
            return "swimming"
        if n in ("person_in_water", "person-in-water", "personwater", "in_water", "in-water"):
            return "person_in_water"

        # COCO person -> person_in_water using water-mask heuristic
        if n == "person" and self.enable_water_heuristic:
            if water_ratio >= self.water_ratio_thres and conf >= self.conf_thres_water:
                return "person_in_water"

        return None

    def _collect_best_by_label(self, detections_np, water_mask=None):
        """Return best detection per label: label -> (x1,y1,x2,y2,conf,raw_name,water_ratio)."""
        if detections_np is None or len(detections_np) == 0:
            return {}

        best = {}
        for det in detections_np:
            x1, y1, x2, y2, conf, cls = det
            conf = float(conf)
            if conf < min(self.conf_thres, self.conf_thres_water):
                continue

            cls_i = int(cls)
            try:
                raw_name = self.yolo.names[cls_i]
            except Exception:
                raw_name = str(cls_i)

            # compute water ratio only when needed
            water_ratio = 0.0
            if water_mask is not None and str(raw_name).strip().lower() in ("person", "person_in_water", "person-in-water", "in_water", "in-water"):
                water_ratio = self._bbox_water_ratio(water_mask, x1, y1, x2, y2)

            label = self._map_raw_to_label(raw_name, conf, water_ratio)

            # if your custom model already outputs person_in_water/drowning/swimming, let it pass
            if label is None:
                # allow raw_name==label directly (case-insensitive)
                rn = str(raw_name).strip().lower()
                if rn in ("drowning", "swimming", "person_in_water"):
                    label = rn

            if label is None:
                continue
            if self.target_labels and (label not in self.target_labels):
                continue

            prev = best.get(label)
            if prev is None or conf > prev[4]:
                best[label] = (int(x1), int(y1), int(x2), int(y2), conf, str(raw_name), float(water_ratio))
        return best

    def _update_track(self, label, bbox, conf, now):
        st = self._tracks.get(label)
        if st is None:
            st = {"count": 0, "last_ts": 0.0, "last_bbox": None, "ema_conf": 0.0}
            self._tracks[label] = st

        # reset if timeout
        if st["last_ts"] and (now - st["last_ts"] > self.track_timeout_s):
            st["count"] = 0
            st["last_bbox"] = None
            st["ema_conf"] = 0.0

        st["last_ts"] = now
        st["last_bbox"] = bbox
        st["ema_conf"] = 0.7 * st["ema_conf"] + 0.3 * float(conf)
        st["count"] += 1
        return st

    def _should_publish(self, label, track_state):
        if label == "drowning":
            return track_state["count"] >= self.confirm_frames_drowning
        if label == "person_in_water":
            return track_state["count"] >= self.confirm_frames_person_in_water
        if label == "swimming":
            return True
        return True
    # ========== Draw center frame ============
    @staticmethod
    def _center_of_bbox(x1, y1, x2, y2):
        """Return bbox center (cx, cy) in pixel coordinates."""
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)
        return cx, cy

    @staticmethod
    def _center_of_box(x1, y1, x2, y2):
        """Backward-compatible alias."""
        return DroneNode._center_of_bbox(x1, y1, x2, y2)
    @staticmethod
    def _draw_cross(img, cx, cy, color=(0, 255, 0), size=10, thickness=2):
        cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
        cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)

    def msg_receiver(self, message):
        global time_last

        if time.time() - time_last < time_to_wait:
            return
        time_last = time.time()

        try:
            cv_image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            # ==== Frame center ====
            h, w = cv_image.shape[:2]
            frame_cx, frame_cy = (w // 2, h // 2)
            self.last_frame_center = (frame_cx, frame_cy)
            #draw camera center crosshair
            self._draw_cross(cv_image, frame_cx, frame_cy, color=(255, 0, 0), size=15, thickness=2)
            cv2.putText(cv_image, f"CAM_C=({frame_cx},{frame_cy})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)            # Improve contrast first (helps water scenes)
            img_infer = self._preprocess(cv_image)

            water_mask = None
            if self.enable_water_heuristic:
                try:
                    water_mask = self._water_mask(img_infer)
                except Exception:
                    water_mask = None

            results = self.yolo(img_infer)
            det = None
            try:
                det = results.xyxy[0].detach().cpu().numpy()
            except Exception:
                if hasattr(results, "pred") and len(results.pred) > 0:
                    det = results.pred[0].detach().cpu().numpy()

            best_by_label = self._collect_best_by_label(det, water_mask=water_mask)

            # annotate + update tracks
            now = time.time()
            colors = {
                "drowning": (0, 0, 255),
                "person_in_water": (0, 0, 255),
                "swimming": (0, 165, 255),
            }

            publish_labels = []
            for label, info in best_by_label.items():
                x1, y1, x2, y2, conf, raw_name, water_ratio = info

                # draw bbox
                col = colors.get(label, (255, 255, 0))
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), col, 2)
                txt = f"{label} {conf:.2f}"
                if label == "person_in_water" and self.enable_water_heuristic and raw_name.strip().lower() == "person":
                    txt += f" water={water_ratio:.2f}"
                cv2.putText(cv_image, txt, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                
                # ===== Add bbox center for person_in_water =====
                if label == "person_in_water":
                    bbox_cx, bbox_cy = self._center_of_bbox(x1, y1, x2, y2)
                    self.last_bbox_center[label] = (bbox_cx, bbox_cy)

                    # offset from camera center (pixel)
                    dx = bbox_cx - frame_cx
                    dy = bbox_cy - frame_cy
                    self.last_center_offset[label] = (dx, dy)


                    # ===== Angle offset (radians) between bbox center and camera center =====
                    # Using your given camera parameters:
                    # horizontal_res=1280, vertical_res=720, horizontal_fov=62.2deg, vertical_fov=48.8deg
                    # Note: positive y_ang means "down" in image coordinates (same as your ArUco snippet).
                    res_w = float(w) if w else float(self.horizontal_res)
                    res_h = float(h) if h else float(self.vertical_res)

                    x_ang = (dx) * self.horizontal_fov / res_w
                    y_ang = (dy) * self.vertical_fov   / res_h

                    self.last_bbox_angles[label] = (x_ang, y_ang)

                    cv2.putText(cv_image,
                                f"x_ang={math.degrees(x_ang):.2f}deg y_ang={math.degrees(y_ang):.2f}deg",
                                (bbox_cx + 10, bbox_cy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    # draw bbox center crosshair + line to camera center
                    self._draw_cross(cv_image, bbox_cx, bbox_cy, color=(0, 0, 255), size=10, thickness=2)
                    cv2.line(cv_image, (frame_cx, frame_cy), (bbox_cx, bbox_cy), (255, 255, 255), 2)

                    # show offset text near bbox
                    cv2.putText(cv_image, f"dx={dx}, dy={dy}",
                                (bbox_cx + 10, bbox_cy + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                tr = self._update_track(label, (x1, y1, x2, y2), conf, now)
                if self._should_publish(label, tr):
                    publish_labels.append((label, conf))
                    # Pause UAV when a "person_in_water"/"drowning" detection is confirmed.
                    # Non-blocking: mission resumes automatically after PERSON_HOLD_SECONDS.
                    try:
                        self.controller.request_pause(reason=label)
                    except Exception:
                        pass

            # send zones to server at interval (can include multiple labels)
            if publish_labels and (now - self.last_send_time >= self.send_interval):
                lat = self.vehicle.location.global_relative_frame.lat
                lon = self.vehicle.location.global_relative_frame.lon
                if lat is not None and lon is not None:
                    radius_m = self.zone_radius_m
                    if self.scale_zone_with_alt:
                        try:
                            alt = self.vehicle.location.global_relative_frame.alt
                            if alt is not None:
                                radius_m = max(self.zone_radius_m, min(60.0, float(alt) * 5.0))
                        except Exception:
                            pass

                    for (label, conf) in publish_labels:
                        self.detected_marker[label] = {
                            "lat": float(lat),
                            "lon": float(lon),
                            "radius_m": float(radius_m),
                            "conf": float(conf),
                            "ts": float(now),
                        }

                    try:
                        self.controller.send_aruco_marker_to_server(self.detected_marker)
                    except Exception as e:
                        print("Error sending zones:", e)

                self.last_send_time = now

            new_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.newimg_pub.publish(new_msg)

            if self.mode == 'land':
                try:
                    if self.vehicle.mode != VehicleMode("LAND"):
                        self.vehicle.mode = VehicleMode("LAND")
                except Exception:
                    pass

        except Exception as e:
            print("YOLOv5 processing error:", e)

# Utility to create controller
_controller = None

def get_controller(connection_str='tcp:127.0.0.1:5763', takeoff_height=3):
    global _controller
    if _controller is None:
        _controller = DroneController(connection_str=connection_str, takeoff_height=takeoff_height)
    return _controller