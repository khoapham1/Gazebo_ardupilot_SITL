import time
import math
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
targetheight = 3
def arm_and_takeoff(targetheight):
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode != "GUIDED":
        print("Waiting to mode GUIDED")
        time.sleep(1)
    ### Armed
    vehicle.armed = True
    while not vehicle.armed:
        print("Waiting to arm")
        time.sleep(1)
    ## Takeoff 
    vehicle.simple_takeoff(targetheight)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"Atitude current: {alt}")
        if alt > targetheight *0.95:
            print("Reached target atitude")
            break
        time.sleep(1)
        print("Takeoff Complete")

def land():
    vehicle.mode = VehicleMode("LAND")
    while not vehicle.armed:
        print("Waiting to land")
        time.sleep(1)
    print("Landed")


if __name__ == '__main__':
    vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True)
    arm_and_takeoff(targetheight)
    time.sleep(2)
    land()