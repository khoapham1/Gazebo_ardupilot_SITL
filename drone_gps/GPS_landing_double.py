import time
import math
from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
from pymavlink import mavutil
import cv2
import numpy as np
import sys

###### GPS Dropoff #######
lat_taco = 10.8502207
lon_taco = 106.7709699

###### GPS HOME #######
#lat_home = 10.8534766
#lon_home = 106.7722452
###Taco Delivery Sevro
time_to_sleep = 10 
servo = 9
dropoff_pwm = 1900 ## PWM that will drop
holdon_pwm = 1100 ##PWM taht holds on to packages



ids_to_find = [1,2]
marker_sizes = [39,4] #cm
marker_heights = [7,3] #meter
takeoff_height = 7
velocity =  .5

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

##Camera
horizontal_res = 640
vertical_res = 480

pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! queue ! videoconvert ! video/x-raw, format=BGR ! queue ! appsink drop=true sync=false"
camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

horizontal_fov = 62.2 * (math.pi / 180)
vertical_fov = 48.8 * (math.pi / 180)

calib_path = "/home/pi/Desktop/Quancool/"
cameraMatrix = np.load(calib_path + 'camera_matrix_gpt.npy')
cameraDistortion = np.load(calib_path + 'dist_coeff_gpt.npy')

###= Counter and script 
found_count = 0
notfound_count = 0
first_run = 0
start_time = 0
end_time = 0
script_mode = 1
ready_to_land = 0
manualArm = True

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('GPS_Landing_double_3.avi',fourcc, 30.0, (horizontal_res, vertical_res))
if not out.isOpened():
    print("Khong the mo VideoWrite!")
    sys.exit()
viewVideo = True


def arm_and_takeoff(targetHeight):
    print("Arming Motor")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.is_armable:
        print("Waitinf for vehicle to become armable")
        time.sleep(1)

    while vehicle.armed:
        print("Waitinf for arming")
        time.sleep(1)

    print(f"Taking off to {targetHeight} meters...")
    vehicle.simple_takeoff(targetHeight)

    while True:
        altitude = vehicle.rangefinder.distance
        print(f"Current Altitude: {altitude:.2f} meters")
        if altitude >= 0.85 * targetHeight:
            break
        time.sleep(1)
    print("Target altitude reached!")
    return None

def send_local_ned_velocity(vx, vy, vz):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def send_land_message(x, y):
    msg = vehicle.message_factory.landing_target_encode(
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        x,
        y,
        0,
        0,
        0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def controlServo(servo_number,pwm_value):
    msg = vehicle.message_factory.command_long_encode(
            0,
            0,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            servo_number,
            pwm_value,
            0,
            0,
            0,
            0,
            0)
    vehicle.send_mavlink(msg)

def get_distance_meter(targetLocation, currentLocation):
    dLat = targetLocation.lat - currentLocation.lat
    dLon = targetLocation.lon - currentLocation.lon
    return math.sqrt((dLon*dLon)+(dLat*dLat))*1.1131195e5

def goto(targetLocation):
    distanceToTargetLocation = get_distance_meter(targetLocation, vehicle.location.global_relative_frame)

    vehicle.simple_goto(targetLocation)

    while vehicle.mode.name=="GUIDED":
        currenrtDistance = get_distance_meter(targetLocation,vehicle.location.global_relative_frame)
        if currenrtDistance < distanceToTargetLocation*0.02:
            print("Reached target waypoint.")
            time.sleep(2)
            break
        time.sleep(2)
    return None


def lander():
    global first_run, notfound_count, found_count, start_time
    if first_run == 0:
        print("Initializing Landing system")
        first_run = 1
        start_time = time.time()
    
    id_found = 0 

    ##Capture frame from Camera
    ret, frame = camera.read()
    frame = cv2.resize(frame,(horizontal_res,vertical_res))

    frame_np = np.array(frame)

    gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)

    ids=''
    corners, ids, rejected = cv2.aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)
    if vehicle.mode != 'LAND':
        vehicle.mode = VehicleMode("LAND")
        while vehicle.mode != 'LAND':
            print('Waiting for drone to ender land mode')
            time.sleep(1) 

    counter = 0 
    corners_np = np.array(corners)

    id_to_find = 0
    marker_height = 0
    altitude = vehicle.rangefinder.distance
    
    if  altitude > marker_heights[1]:
        id_to_find = ids_to_find[0]
        marker_height =  marker_heights[0]
        marker_size = marker_sizes[0]
    elif altitude < marker_heights[1]:
        id_to_find = ids_to_find[1]
        marker_height =  marker_heights[1]
        marker_size = marker_sizes[1]
    
    print("Marker "+str(id_to_find)+"at "+str(marker_size)+" cms.")

    try:
        if ids is not None:
            for id in ids:
                if id == id_to_find:
                    corners_single = [corners[counter]]
                    corners_single_np =  np.asarray(corners_single)
        
                    ret = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, cameraMatrix, cameraDistortion)
                    rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
                    #x, y, z = tvec[0], tvec[1], tvec[2]
                    x = '{:.2f}'.format(tvec[0])
                    y = '{:.2f}'.format(tvec[1])
                    z = '{:.2f}'.format(tvec[2])

                    x_sum = 0
                    y_sum = 0

                    x_sum = corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]
                    y_sum = corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]

                    # Calculate center coordinates
                    x_avg = x_sum*0.25
                    y_avg = y_sum*0.25
                    
                    x_ang = (x_avg - horizontal_res*0.5) * (horizontal_fov/horizontal_res)
                    y_ang = (y_avg - vertical_res*0.5) * (vertical_fov/vertical_res)

                    if vehicle.mode != 'LAND':
                        vehicle.mode = VehicleMode("LAND")
                        while vehicle.mode != 'LAND':                            
                            time.sleep(1)
                        print("-------------------------")
                        print("Vehicle now in LAND mode")
                        send_land_message(x_ang, y_ang)
                    else:
                        send_land_message(x_ang,y_ang)
                        pass
                    cv2.drawFrameAxes(frame_np, cameraMatrix, cameraDistortion, rvec, tvec, 10)
                    print("X Center pixel: "+str(x_avg)+"Y Center pixel: "+str(y_avg))
                    print("Found count: "+str(found_count)+ "Notfound count: "+str(notfound_count))
                    print(f"Marker Position: x={x:.2f} y={y:.2f} z={z:.2f}")
                    found_count += 1
                    print("")
                counter+=1
            if id_found==0:
                notfound_count+=1                    
        else:
            notfound_count += 1           
    except Exception as e:
        print(f'Detection error: {str(e)}')
        notfound_count += 1
    
    out.write(frame_np)

    cv2.imshow("Frame", frame_np)
    cv2.waitKey(1)

####################### MAIN ###########################
if __name__ == "__main__":
    vehicle = connect('/dev/ttyAMA2',baud=921600,wait_ready=True)
    
    # Configure precision landing parameters
    vehicle.parameters['PLND_ENABLED'] = 1
    vehicle.parameters['PLND_TYPE'] = 1
    vehicle.parameters['PLND_EST_TYPE'] = 0
    vehicle.parameters['LAND_SPEED'] = 20

    lat_home = vehicle.location.global_relative_frame.lat
    lon_home = vehicle.location.global_relative_frame.lon


    wp_home = LocationGlobalRelative(lat_home,lon_home, takeoff_height)## waitpoint Home    
    wp_taco = LocationGlobalRelative(lat_taco,lon_taco, takeoff_height)## waitpoint taco

    distanceBetweenLaunchAndtaco = get_distance_meter(wp_taco, wp_home)
    print("Taco dropoff is "+str(distanceBetweenLaunchAndtaco)+" meters from home wp.")

    #controlServo(servo,holdon_pwm)
    time.sleep(1)
    ######FLY Drone to taco DropOff ########
    arm_and_takeoff(takeoff_height)
    goto(wp_taco)
    while vehicle.armed:
        lander()
    print("")
    print("----------------------------------")
    print("Arrived at the taco destination!")
    print("Dropping tacos and heading home.")
    print("----------ENJOY!----------------")
    ### DROP ###
    # controlServo(servo,dropoff_pwm)
    # time.sleep(1)
    # time.sleep(time_to_sleep)

    # #### FLY the drone to home wayponit ###

    # arm_and_takeoff(takeoff_height)
    # goto(wp_home)

    # while vehicle.armed:
    #     lander()
    # print("")
    # print("----------------------------------")
    # print("Made it home for another delivery!")
    # print("----------------------------------")

    # #### HOlD servo ###
    # controlServo(servo,holdon_pwm)
    # time.sleep(1)
    # time.sleep(time_to_sleep)

    out.release()
    camera.release()
    cv2.destroyAllWindows()    







