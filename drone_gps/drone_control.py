
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

# ---- camera / aruco params (copy from yours) ---
aruco = cv2.aruco
bridge = CvBridge()
ids_to_find = [1, 2]


# find_aruco giờ là động, sẽ set qua class
marker_sizes = [60, 10]          # cm
marker_heights = [10, 3]         # m, altitude thresholds
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()


parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.05
if hasattr(cv2.aruco, 'CORNER_REFINE_SUBPIX'):
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.01

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
        """
        Send detected ArUco markers to server endpoint.
        """
        try:
            response = requests.post('http://localhost:5000/update_aruco_markers', json={'markers': markers}, timeout=2)
            if response.status_code == 200:
                print("Successfully sent ArUco markers to server")
            else:
                print(f"Failed to send ArUco markers, status code: {response.status_code}") 
        except Exception as e:
            print(f"Error sending ArUco markers to server: {e}")
    # MAVLink helpers
    def send_local_ned_velocity(self, vx, vy, vz):
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
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

    def goto(self, targetLocation, tolerance=2.0, timeout=60, speed=2.5):
        """
        Goto with increased tolerance and timeout to avoid stuck, record position.
        """
        distanceToTargetLocation = self.get_distance_meters(targetLocation, self.vehicle.location.global_relative_frame)
        self.set_speed(speed)
        self.vehicle.simple_goto(targetLocation, groundspeed=speed)
        start_dist = distanceToTargetLocation
        start_time = time.time()
        while self.vehicle.mode.name == "GUIDED" and time.time() - start_time < timeout:
            currentDistance = self.get_distance_meters(targetLocation, self.vehicle.location.global_relative_frame)
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
        print("Started scanning for ArUco markers 3-17")

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
        Fly to waypoints while scanning ArUco 3-17, print detected positions, return via interpolated recorded path, precision land at home.
        """
        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        # Clear previous flown path
        self.flown_path = []

        # Takeoff from home
        print("Arming and taking off")
        self.arm_and_takeoff(loiter_alt)
        time.sleep(1)

        self.start_scanning()

        # Store home
        home_lat = self.vehicle.location.global_relative_frame.lat
        home_lon = self.vehicle.location.global_relative_frame.lon
        wp_home = LocationGlobalRelative(home_lat, home_lon, loiter_alt)
        print(f"Home recorded at lat={home_lat:.6f}, lon={home_lon:.6f}")

        for i, wp in enumerate(waypoints[1:-1]):
            wp_loc = LocationGlobalRelative(wp[0], wp[1], loiter_alt)
            print(f"Flying to waypoint {i+1}: {wp[0]}, {wp[1]}")
            self.goto(wp_loc)

        goal_wp = waypoints[-1]
        wp_target = LocationGlobalRelative(goal_wp[0], goal_wp[1], loiter_alt)
        print("Flying to final target", goal_wp[0], goal_wp[1])
        self.goto(wp_target)

        # Stop scanning and print detected markers
        print("Detected ArUco markers (3-17):")
        if self.ros_node and self.ros_node.detected_marker:
            for marker_id, pos in sorted(self.ros_node.detected_marker.items()):
                print(f"ID {marker_id}: lat={pos[0]:.6f}, lon={pos[1]:.6f}")
        else:
            print("No markers detected")
        self.stop_scanning()
        self._start_aruco_processing(duration=30, mode='land')

        # Interpolate recorded path for return
        return_path = self.interpolate_path(self.flown_path[::-1], num_points=20)  # Reverse and interpolate
        print("Interpolated return path:", return_path)
        self._start_aruco_processing(duration=30, mode='land')
        # Fly back via interpolated return path
        print("Returning to home via recorded path")


    def fly_selected_points(self, waypoints, loiter_alt=3):
        """
        Fly through selected points without landing at each point.
        Only land at the final point and return home.
        """
        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        # Clear previous flown path
        self.flown_path = []

        # Takeoff from home
        print("Arming and taking off for selected points mission")
        self.arm_and_takeoff(loiter_alt)
        time.sleep(1)

        # Store home
        home_lat = self.vehicle.location.global_relative_frame.lat
        home_lon = self.vehicle.location.global_relative_frame.lon
        wp_home = LocationGlobalRelative(home_lat, home_lon, loiter_alt)
        print(f"Home recorded at lat={home_lat:.6f}, lon={home_lon:.6f}")

        # Fly to all selected waypoints
        for i, wp in enumerate(waypoints[1:]):
            wp_loc = LocationGlobalRelative(wp[0], wp[1], loiter_alt)
            print(f"Flying to waypoint {i+1}: {wp[0]}, {wp[1]}")
            self.goto(wp_loc)

        # Return to home after visiting all points
        print("Returning to home")
        self.goto(wp_home)

        # Land at home
        print("Landing at home")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode != "LAND":
            print("Waiting for LAND mode...")
            time.sleep(1)
        print("Starting precision landing phase at home (aruco)...")
        self._start_aruco_processing(duration=30, mode='land')
        # Wait until disarmed
        while self.vehicle.armed:
            print("Waiting for disarming...")
            time.sleep(1)
            
        print("Selected points mission complete")

class DroneNode(Node):
    def __init__(self, controller, mode='land'):
        node_name = f"drone_node_for_aruco{int(time.time()*1000)}"
        super().__init__(node_name)
        self.controller = controller  # Store the controller object
        self.vehicle = controller.vehicle  # Access the vehicle from the controller
        self.mode = mode
        self.newimg_pub = self.create_publisher(Image, '/UAV/forward/image_new', 10)
        self.subscription = self.create_subscription(Image, '/UAV/forward/image_raw', self.msg_receiver, 10)
        # tracking state (we keep simple smoothing but no lost-handling logic)
        self.last_detection_time = 0.0
        self.detected_marker = {}
        # image preprocessing helpers
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.last_marker_send_time = 0.0
        self.marker_send_interval = 2.0  # seconds
        # Lấy find_aruco từ controller (động)
        self.find_aruco = self.controller.find_aruco

    def preprocess(self, gray_img):
        """
        Preprocess image to improve ArUco detectability:
         - CLAHE (local histogram equalization)
         - small gaussian blur
        """
        try:
            clahe_img = self.clahe.apply(gray_img)
        except Exception:
            clahe_img = gray_img

        blur = cv2.GaussianBlur(clahe_img, (3,3), 0)
        return blur

    def msg_receiver(self, message):
        global time_last
        # throttle processing
        if time.time() - time_last < time_to_wait:
            return
        time_last = time.time()

        try:
            cv_image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Preprocess to enhance marker contrast
            preproc = self.preprocess(gray_img)

            # detect markers on preprocessed image first
            corners, ids, rejected = aruco.detectMarkers(preproc, aruco_dict, parameters=parameters)

            id_to_find = None
            marker_size = None
            found_land = False

            if self.mode == 'land':
                if self.vehicle.mode != 'LAND':
                    self.vehicle.mode = VehicleMode("LAND")
                    while self.vehicle.mode != 'LAND':
                        print('Waiting for drone to enter land mode')
                        time.sleep(1)
                altitude = self.vehicle.location.global_relative_frame.alt
                if altitude is None:
                    altitude = 0.0
                if altitude > marker_heights[1]:
                    id_to_find = ids_to_find[0]
                    marker_size = marker_sizes[0]
                else:
                    id_to_find = ids_to_find[1]
                    marker_size = marker_sizes[1]

            try:
                if ids is not None:
                    ids_flat = ids.flatten()
                    for idx, marker_id in enumerate(ids_flat):
                        marker_id = int(marker_id)
                        if self.mode == 'land' and marker_id == id_to_find:
                            # we found the marker we want
                            corners_single = [corners[idx]]
                            corners_single_np = np.asarray(corners_single)

                            # estimate pose (size in meters)
                            marker_size_m = marker_size / 100.0
                            ret = aruco.estimatePoseSingleMarkers(corners, marker_size,
                                                                  cameraMatrix=np_camera_matrix,
                                                                  distCoeffs=np_dist_coeff)
                            rvec, tvec = ret[0][idx][0, :], ret[1][idx][0, :]

                            x = float(tvec[0])
                            y = float(tvec[1])
                            z = float(tvec[2])

                            x_sum = 0
                            y_sum = 0

                            x_sum = corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]
                            y_sum = corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]


                            x_avg = x_sum * 0.25
                            y_avg = y_sum * 0.25

                            x_ang = (x_avg - horizontal_res * 0.5) * horizontal_fov / horizontal_res
                            y_ang = (y_avg - vertical_res * 0.5) * vertical_fov / vertical_res

                            self.controller.send_land_message(x_ang, y_ang)
                            print(f"Sending landing target x_ang={x_ang:.3f}, y_ang={y_ang:.3f}")

                            # annotate for visualization
                            marker_position = f'MARKER POS: x={x:.2f} y={y:.2f} z={z:.2f}'
                            try:
                                cv2.drawFrameAxes(cv_image, np_camera_matrix, np_dist_coeff, rvec, tvec, 0.1)
                            except Exception:
                                pass
                            cv2.putText(cv_image, marker_position, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), thickness=2)
                            print(marker_position)

                            self.last_detection_time = time.time()
                            found_land = True

                        if marker_id in self.find_aruco:  # Sử dụng self.find_aruco động
                            marker_size_2 = 60
                            ret = aruco.estimatePoseSingleMarkers(corners, marker_size_2,
                                                                  cameraMatrix=np_camera_matrix,
                                                                  distCoeffs=np_dist_coeff)
                            rvec, tvec = ret[0][idx][0, :], ret[1][idx][0, :]

                            x = float(tvec[0])
                            y = float(tvec[1])
                            z = float(tvec[2])
                            marker_position = f'MARKER POS: x={x:.2f} y={y:.2f} z={z:.2f}'
                            print(marker_position)
                            if -10 < y < 15:
                                lat = self.vehicle.location.global_relative_frame.lat
                                lon = self.vehicle.location.global_relative_frame.lon
                                self.detected_marker[marker_id] = [lat, lon]
                                print(f"Detected marker ID {marker_id} at lat={lat:.6f}, lon={lon:.6f}")
                                try:
                                    self.controller.send_aruco_marker_to_server(self.detected_marker)
                                except Exception as e:
                                    print("Error sending ArUco marker:", e)

                            
                # Draw all detected markers
                if ids is not None and len(corners) > 0:
                    for i, corner in enumerate(corners):
                        # Vẽ khung marker
                        aruco.drawDetectedMarkers(cv_image, [corner])

                        c = corner[0]
                        top_left = tuple(c[0].astype(int))
                        marker_id = int(ids[i])

                        cv2.putText(cv_image, f"ID {marker_id}", (top_left[0], top_left[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)


            except Exception as e:
                print("Pose estimation error:", e)

            new_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.newimg_pub.publish(new_msg)

        except Exception as e:
            # swallow for safety but print
            print("ArUco processing error:", e)


# Utility to create controller
_controller = None

def get_controller(connection_str='tcp:127.0.0.1:5763', takeoff_height=3):
    global _controller
    if _controller is None:
        _controller = DroneController(connection_str=connection_str, takeoff_height=takeoff_height)
    return _controller