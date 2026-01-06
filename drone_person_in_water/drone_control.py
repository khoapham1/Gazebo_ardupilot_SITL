#!/usr/bin/env python3
# drone_control.py (thay thế ArUco bằng YOLOv5)
import time
import math
import threading
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
import torch
from torchvision import transforms

# ---- YOLOv5 Parameters ----
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng tin cậy
PERSON_CLASS_ID = 0  # ID của class "person" trong COCO dataset
DETECTION_AREA_THRESHOLD = 0.1  # Ngưỡng diện tích bounding box

# Load YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = CONFIDENCE_THRESHOLD
    print("YOLOv5 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    model = None

# Camera parameters
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

# ---- Streaming Image ----
def start_image_streamer(topic_name='/UAV/forward/image_new'):
    """
    Start background ROS2 node that listens to camera topic and updates latest JPEG.
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
    """
    Return lastest JPEG bytes or None.
    """
    global _latest_frame_jpeg, _latest_frame_lock
    with _latest_frame_lock:
        return _latest_frame_jpeg

# ---- DroneController class ----
class DroneController:
    def __init__(self, connection_str='tcp:127.0.0.1:5763', takeoff_height=3):
        self.connection_str = connection_str
        print("Connecting to vehicle on", connection_str)
        self.vehicle = connect(connection_str, wait_ready=True, timeout=120)
        
        # Set landing parameters
        try:
            self.vehicle.parameters['PLND_ENABLED'] = 1
            self.vehicle.parameters['PLND_TYPE'] = 1
            self.vehicle.parameters['PLND_EST_TYPE'] = 0
            self.vehicle.parameters['LAND_SPEED'] = 20
        except Exception:
            print("Failed to set some landing parameters")
        
        self.takeoff_height = takeoff_height
        if not rclpy.ok():
            rclpy.init(args=None)
        self.flown_path = []
        self.ros_node = None
        self.executor = None
        self.scan_thread = None
        self.scan_running = False
        # Lưu trữ các phát hiện person in water
        self.person_detections = {}
    
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
    
    def send_person_detection_to_server(self, detection):
        """Gửi thông tin phát hiện person in water đến server"""
        try:
            response = requests.post('http://localhost:5000/update_person_detection', 
                                   json=detection, 
                                   timeout=2)
            if response.status_code == 200:
                print("Successfully sent person detection to server")
            else:
                print(f"Failed to send detection, status code: {response.status_code}")
        except Exception as e:
            print(f"Error sending person detection to server: {e}")
    
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
            0.0,
            0
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
            if alt >= 0.95 * targetHeight:
                break
            time.sleep(1)
        print("Reached takeoff altitude")
        return None

    def start_person_detection(self):
        """Bắt đầu phát hiện person in water"""
        if self.scan_running:
            print("Person detection already running")
            return
        self.ros_node = DroneNode(self, mode='person_detection')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ros_node)
        self.scan_running = True
        self.scan_thread = threading.Thread(target=self._scan_loop)
        self.scan_thread.start()
        print("Started person in water detection")

    def _scan_loop(self):
        while rclpy.ok() and self.scan_running and self.vehicle.armed:
            self.executor.spin_once(timeout_sec=0.1)

    def stop_person_detection(self):
        """Dừng phát hiện person in water"""
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
        print("Stopped person detection")

    def fly_waypoints_with_person_detection(self, waypoints, loiter_alt=3):
        """
        Bay theo waypoints và phát hiện person in water
        """
        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        # Clear previous flown path
        self.flown_path = []

        # Takeoff
        print("Arming and taking off")
        self.arm_and_takeoff(loiter_alt)
        time.sleep(1)

        # Bắt đầu phát hiện person in water
        self.start_person_detection()

        # Fly to waypoints
        for i, wp in enumerate(waypoints[1:]):
            wp_loc = LocationGlobalRelative(wp[0], wp[1], loiter_alt)
            print(f"Flying to waypoint {i+1}: {wp[0]}, {wp[1]}")
            self.goto(wp_loc)
            time.sleep(1)

        # Dừng phát hiện
        self.stop_person_detection()

        # Return to home
        home_lat = self.vehicle.location.global_relative_frame.lat
        home_lon = self.vehicle.location.global_relative_frame.lon
        wp_home = LocationGlobalRelative(home_lat, home_lon, loiter_alt)
        print("Returning to home")
        self.goto(wp_home)

        # Land
        print("Landing")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.armed:
            time.sleep(1)

        print("Mission complete")

class DroneNode(Node):
    def __init__(self, controller, mode='person_detection'):
        node_name = f"drone_node_yolo_{int(time.time()*1000)}"
        super().__init__(node_name)
        self.controller = controller
        self.vehicle = controller.vehicle
        self.mode = mode
        self.subscription = self.create_subscription(Image, '/UAV/forward/image_raw', self.msg_receiver, 10)
        self.bridge = CvBridge()
        self.last_detection_time = 0.0
        self.detection_interval = 2.0  # Gửi detection mỗi 2 giây
        self.person_detections = {}

    def calculate_position_from_bbox(self, bbox, altitude):
        """
        Tính toán vị trí địa lý từ bounding box và độ cao
        bbox: [x1, y1, x2, y2] trong pixel
        altitude: độ cao hiện tại (m)
        """
        # Tính center của bounding box
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        # Chuyển đổi pixel sang góc
        x_ang = (cx - horizontal_res * 0.5) * horizontal_fov / horizontal_res
        y_ang = (cy - vertical_res * 0.5) * vertical_fov / vertical_res
        
        # Tính khoảng cách từ drone đến object (ước lượng)
        distance = altitude * math.tan(abs(y_ang))
        
        # Lấy yaw hiện tại của drone
        try:
            yaw = self.vehicle.attitude.yaw
        except:
            yaw = 0
        
        # Tính offset theo hướng yaw
        north_offset = distance * math.cos(yaw)
        east_offset = distance * math.sin(yaw)
        
        # Chuyển đổi sang lat/lon
        lat_offset = north_offset / 111111.0  # 1 độ ~ 111,111m
        lon_offset = east_offset / (111111.0 * math.cos(math.radians(self.vehicle.location.global_frame.lat)))
        
        # Tính vị trí thực tế
        lat = self.vehicle.location.global_frame.lat + lat_offset
        lon = self.vehicle.location.global_frame.lon + lon_offset
        
        # Tính kích thước vùng (radius dựa trên bounding box)
        bbox_width = abs(bbox[2] - bbox[0])
        bbox_height = abs(bbox[3] - bbox[1])
        area_percent = (bbox_width * bbox_height) / (horizontal_res * vertical_res)
        radius_meters = 5 + (area_percent * 20)  # Radius từ 5-25m
        
        return lat, lon, radius_meters

    def msg_receiver(self, message):
        global time_last
        if time.time() - time_last < time_to_wait:
            return
        time_last = time.time()

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            
            # Run YOLOv5 inference
            if model is not None:
                results = model(cv_image)
                
                # Parse results
                detections = results.pandas().xyxy[0]
                
                # Filter for person detections
                person_detections = detections[detections['class'] == PERSON_CLASS_ID]
                
                # Get current altitude
                altitude = self.vehicle.location.global_relative_frame.alt
                if altitude is None:
                    altitude = 0.0
                
                # Process each person detection
                for _, detection in person_detections.iterrows():
                    confidence = detection['confidence']
                    bbox = [detection['xmin'], detection['ymin'], 
                           detection['xmax'], detection['ymax']]
                    
                    # Vẽ bounding box
                    cv2.rectangle(cv_image, 
                                 (int(bbox[0]), int(bbox[1])), 
                                 (int(bbox[2]), int(bbox[3])), 
                                 (0, 0, 255), 3)  # Red color
                    
                    # Vẽ text "PERSON IN WATER"
                    cv2.putText(cv_image, f"PERSON IN WATER {confidence:.2f}",
                               (int(bbox[0]), int(bbox[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Tính toán vị trí nếu có độ cao hợp lệ
                    if altitude > 1.0:  # Chỉ tính nếu độ cao > 1m
                        lat, lon, radius = self.calculate_position_from_bbox(bbox, altitude)
                        
                        # Tạo detection ID dựa trên timestamp và bbox
                        detection_id = f"person_{int(time.time())}_{int(bbox[0])}"
                        
                        # Lưu detection
                        detection_data = {
                            'id': detection_id,
                            'lat': lat,
                            'lon': lon,
                            'radius': radius,
                            'confidence': float(confidence),
                            'timestamp': time.time(),
                            'altitude': float(altitude)
                        }
                        
                        self.person_detections[detection_id] = detection_data
                        
                        # Gửi lên server mỗi interval
                        if time.time() - self.last_detection_time > self.detection_interval:
                            self.controller.send_person_detection_to_server(detection_data)
                            self.last_detection_time = time.time()
                            print(f"Detected person in water at lat={lat:.6f}, lon={lon:.6f}, radius={radius:.1f}m")
                    
                # Nếu không phát hiện người, ghi thông tin
                if len(person_detections) == 0:
                    cv2.putText(cv_image, "No person detected",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert back to ROS message if needed
            # (có thể publish hình ảnh đã được annotate nếu cần)
            
        except Exception as e:
            print(f"YOLO processing error: {e}")

# Utility to create controller
_controller = None

def get_controller(connection_str='tcp:127.0.0.1:5763', takeoff_height=3):
    global _controller
    if _controller is None:
        _controller = DroneController(connection_str=connection_str, takeoff_height=takeoff_height)
    return _controller