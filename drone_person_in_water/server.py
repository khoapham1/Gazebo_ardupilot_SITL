# server.py (updated to support YOLOv5 person in water detection)
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import threading, time
from drone_control import get_controller, get_lastest_frame
import json
from planner import run_planner
from dronekit import VehicleMode, LocationGlobalRelative
import math
import cv2
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create/connect controller singleton
controller = get_controller(connection_str='tcp:127.0.0.1:5763', takeoff_height=3)

# Lưu trữ phát hiện person in water
person_detections = {}
prev_loc = None
distance_traveled = 0.0

# Start image streamer
try:
    controller.start_image_stream(topic_name='/UAV/forward/image_new')
except Exception as e:
    print("Warning: Failed to start image streamer:", e)

def mjpeg_generator():
    """Generator that yields MJPEG frames."""
    while True:
        frame = get_lastest_frame()
        if frame is None:
            placeholder = cv2.imencode('.jpg', np.zeros((1,1,3), np.uint8))[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def telemetry_loop():
    global prev_loc, distance_traveled
    while True:
        try:
            v = controller.vehicle
            loc = v.location.global_frame
            if loc.lat is None or loc.lon is None:
                time.sleep(0.5)
                continue
            current_loc = (loc.lat, loc.lon)
            velocity = v.groundspeed if v.groundspeed else 0.0
            if prev_loc:
                dlat = current_loc[0] - prev_loc[0]
                dlon = current_loc[1] - prev_loc[1]
                dist_step = math.sqrt((dlat**2) + (dlon**2)) * 1.113195e5
                distance_traveled += dist_step
            prev_loc = current_loc
            data = {
                'lat': loc.lat,
                'lon': loc.lon,
                'alt': v.location.global_relative_frame.alt,
                'mode': str(v.mode.name),
                'velocity': velocity,
                'distance_traveled': distance_traveled
            }
            socketio.emit('telemetry', data)
        except Exception as e:
            socketio.emit('telemetry', {'error': str(e)})
        time.sleep(0.5)

# Endpoint để cập nhật person detection
@app.route('/update_person_detection', methods=['POST'])
def update_person_detection():
    try:
        payload = request.get_json(silent=True) or {}
        
        if not payload:
            return jsonify({'error': 'No data provided'}), 400
        
        detection_id = payload.get('id')
        if not detection_id:
            return jsonify({'error': 'Missing detection ID'}), 400
        
        global person_detections
        person_detections[detection_id] = payload
        
        print(f"Received person detection: {detection_id}")
        
        # Gửi realtime đến tất cả clients
        socketio.emit('person_detection_update', {'detection': payload})
        
        return jsonify({'status': 'success', 'detection_id': detection_id})
    except Exception as e:
        print(f"Error updating person detection: {e}")
        return jsonify({'error': str(e)}), 500

# Endpoint để lấy danh sách person detections
@app.route('/get_person_detections', methods=['GET'])
def get_person_detections():
    return jsonify(person_detections)

# Endpoint để xóa person detections
@app.route('/clear_person_detections', methods=['POST'])
def clear_person_detections():
    global person_detections
    person_detections = {}
    socketio.emit('person_detections_cleared', {})
    return jsonify({'status': 'cleared'})

# Start person detection
@app.route('/start_person_detection', methods=['POST'])
def start_person_detection():
    try:
        controller.start_person_detection()
        return jsonify({'status': 'success', 'message': 'Person detection started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Stop person detection
@app.route('/stop_person_detection', methods=['POST'])
def stop_person_detection():
    try:
        controller.stop_person_detection()
        return jsonify({'status': 'success', 'message': 'Person detection stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Fly mission with person detection
@app.route('/fly_with_person_detection', methods=['POST'])
def fly_with_person_detection():
    global distance_traveled
    distance_traveled = 0.0
    
    try:
        payload = request.get_json(silent=True) or {}
        points = payload.get('points', [])
        
        if not points:
            return jsonify({'error': 'No points selected'}), 400
        
        # Lấy home position làm start
        v = controller.vehicle
        start = [v.location.global_frame.lat, v.location.global_frame.lon]
        
        # Tạo waypoints
        waypoints = [start]
        for point in points:
            waypoints.append([point['lat'], point['lon']])
        
        # Gửi planned path đến web
        socketio.emit('planned_path', {'waypoints': waypoints})
        
        # Start mission với person detection
        def mission():
            try:
                socketio.emit('mission_status', {'status': 'starting', 'waypoints': waypoints})
                controller.fly_waypoints_with_person_detection(waypoints, loiter_alt=3)
                socketio.emit('mission_status', {'status': 'completed'})
            except Exception as e:
                socketio.emit('mission_status', {'status': 'error', 'error': str(e)})
        
        t = threading.Thread(target=mission, daemon=True)
        t.start()
        
        return jsonify({'status': 'mission_started', 'waypoints': waypoints})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Các endpoints khác giữ nguyên...
@app.route('/get_gps_stations', methods=['GET'])
def get_gps_stations():
    try:
        with open('file_gps_station.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_mission', methods=['POST'])
def start_mission():
    global distance_traveled
    distance_traveled = 0.0
    
    try:
        payload = request.get_json(silent=True) or {}
        station_name = payload.get('station', 'station1')
        with open('file_gps_station.json', 'r') as f:
            data = json.load(f)

        if station_name not in data:
            return jsonify({'error': f'station "{station_name}" not found in JSON'}), 400

        v = controller.vehicle
        start = [v.location.global_frame.lat, v.location.global_frame.lon]
        
        waypoints = [start]
        for point in data[station_name]:
            waypoints.append([point['lat'], point['lon']])
        
        socketio.emit('planned_path', {'waypoints': waypoints})
        
        # Sử dụng hàm fly_waypoints_with_person_detection
        def mission():
            try:
                socketio.emit('mission_status', {'status': 'starting', 'waypoints': waypoints})
                controller.fly_waypoints_with_person_detection(waypoints, loiter_alt=3)
                socketio.emit('mission_status', {'status': 'completed'})
            except Exception as e:
                socketio.emit('mission_status', {'status': 'error', 'error': str(e)})
        
        t = threading.Thread(target=mission, daemon=True)
        t.start()
        
        return jsonify({'status': 'mission_started', 'station': station_name, 'waypoints': waypoints})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    t = threading.Thread(target=telemetry_loop, daemon=True)
    t.start()
    socketio.run(app, host='0.0.0.0', port=5000)