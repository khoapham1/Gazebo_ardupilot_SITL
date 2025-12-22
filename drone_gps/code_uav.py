import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
altitude_target=10

def arm_and_takeoff(altitude_target):
    vehicle.mode= VehicleMode("GUIDED")
    while vehicle.mode !="GUIDED":
        print("Waiting to mode Guided!")
        time.sleep(1) #mms
    vehicle.armed = True
    while not vehicle.armed:
        print("waiting to arm")
        time.sleep(1)
    print("Armed!")
    vehicle.simple_takeoff(altitude_target)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"Altitude: {alt}")
        if alt >= altitude_target * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)
    print("Takeoff complete")
def land():
    print("Landing")
    vehicle.mode = VehicleMode("LAND")
    print("landed")

if __name__ == '__main__':

    vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True)
    vehicle.parameters['PLND_ENABLED'] = 1
    vehicle.parameters['PLND_TYPE'] = 1
    vehicle.parameters['PLND_EST_TYPE'] = 0
    vehicle.parameters['LAND_SPEED'] = 30  # cm/s
    arm_and_takeoff(altitude_target)
    time.sleep(5)
    land()