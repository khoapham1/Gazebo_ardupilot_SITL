import time
import math
import numpy as np

import rclpy 
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil


targetHeight = 5  # m

lat_pkg = 10.85091904
lon_pkg = 106.77125894


def arm_and_takeoff(targetHeight):
    print("Vehicle is now armable")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode != 'GUIDED':
        print("Waiting for GUIDED mode...")
        time.sleep(1)
    print("Vehicle is now in GUIDED mode")

    vehicle.armed = True #drone is armed
    while not vehicle.armed:# Kiem tra drone da duoc arm chua?
        print("Waiting for vehicle to become armed...")
        time.sleep(1)

    print("Armed!") # Arm xong!!

    vehicle.simple_takeoff(targetHeight) # Take off len do cao targetHeight
    while True:
        altitude = vehicle.location.global_relative_frame.alt 
        # altitude = vehicle.location.global_relative_frame.rangefinder_distance
        print(f"Altitude: {altitude:.1f} m")

        if altitude >= targetHeight * 0.95: # Neu do cao lon hon 95% targetHeight
            break
        time.sleep(1)
    
    print("Reached target altitude!")


def get_distance_meters(targetLocation, currentLocation):
    dlat = targetLocation.lat - currentLocation.lat
    dlong = targetLocation.lon - currentLocation.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5 #so meter tuong duong voi 1 do kinh do/vi do(111.3195km)

def goto(targetLocation):
    vehicle.simple_goto(targetLocation)
    while True:
        currentLocation = vehicle.location.global_relative_frame
        distance = get_distance_meters(targetLocation, currentLocation)
        print(f"Distance to target: {distance:.1f} m")
        if distance <= 1.0: # Neu khoang cach den diem dich nho hon 1m
            print("Reached target location")
            break
        time.sleep(2)
    print("Arrived at target location")
    return False
def land():
    print("Landing...")
    vehicle.mode = VehicleMode("LAND")
    
    while vehicle.armed: #cho den khi drone chua duoc disarm
        print(f"Waiting for landing... {vehicle.location.global_relative_frame.alt:.1f} m")
        time.sleep(1)
    print("Landed and disarmed.")

if __name__ == '__main__':
    vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True, timeout=120)


    try:
        vehicle.parameters['LAND_SPEED'] = 25 # cm/s
    except Exception as e:
        print(f"Failed to set landing parameters: {e}")
    lat_home = vehicle.location.global_relative_frame.lat
    lon_home = vehicle.location.global_relative_frame.lon

    wp_home = LocationGlobalRelative(lat_home, lon_home, targetHeight)
    wp_pkg = LocationGlobalRelative(lat_pkg, lon_pkg, targetHeight)

    arm_and_takeoff(targetHeight)
    goto(wp_pkg)
    land()

    print("")
    print("----------------------------------")
    print("Arrived at the taco destination!")
    print("----------ENJOY!----------------")

    arm_and_takeoff(targetHeight)
    goto(wp_home)
    land()
    print("")
    print("----------------------------------")
    print("Going back home")
    print("----------ENJOY!----------------")
    vehicle.close()


    