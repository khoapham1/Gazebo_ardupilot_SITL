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

_latest_frame_lock = threading.Lock()
_latest_frame_jpeg = None


class ImageStreamerNode(Node):
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