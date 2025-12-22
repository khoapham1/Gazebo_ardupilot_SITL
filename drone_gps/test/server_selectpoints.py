#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal demo for teaching (Select Points):
- Leaflet map: click to select multiple GPS points
- Send selected points to server via Flask route (HTTP)
- Server emits planned_path via Socket.IO event (realtime)

Run:
  pip install flask flask-socketio eventlet
  python server_selectpoints.py
Open:
  http://localhost:5000
"""

from __future__ import annotations

import time
from threading import Thread, Lock
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO

APP_HOST = "0.0.0.0"
APP_PORT = 5000

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

_selected_lock = Lock()
selected_points: List[Dict[str, float]] = []  # [{"lat":..., "lon":...}, ...]

def _safe_points_copy() -> List[Dict[str, float]]:
    with _selected_lock:
        return [dict(p) for p in selected_points]

@app.get("/")
def index():
    # Serve the local index_selectpoints.html next to this file
    return send_from_directory(".", "index_selectpoints.html")

@app.post("/fly_selected")
def fly_selected():
    """HTTP route: web bấm nút -> gửi request 1 lần -> server trả JSON.
    Đồng thời server phát planned_path (Socket.IO) để client vẽ đường xanh.
    """
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    points = payload.get("points", [])

    if not isinstance(points, list) or len(points) == 0:
        return jsonify({"ok": False, "error": "No points provided"}), 400

    cleaned: List[Dict[str, float]] = []
    try:
        for p in points:
            lat = float(p["lat"])
            lon = float(p["lon"])
            cleaned.append({"lat": lat, "lon": lon})
    except Exception:
        return jsonify({"ok": False, "error": "Invalid points format"}), 400

    with _selected_lock:
        selected_points.clear()
        selected_points.extend(cleaned)

    waypoints = [[p["lat"], p["lon"]] for p in cleaned]
    socketio.emit("planned_path", {"waypoints": waypoints})

    return jsonify({"ok": True, "count": len(cleaned)})

@app.post("/clear_selected")
def clear_selected():
    """Optional: clear selected points on server."""
    with _selected_lock:
        selected_points.clear()
    socketio.emit("planned_path", {"waypoints": []})
    return jsonify({"ok": True})

@socketio.on("connect")
def on_connect():
    """When a client opens the page, push the last planned path (if any)."""
    pts = _safe_points_copy()
    socketio.emit("planned_path", {"waypoints": [[p["lat"], p["lon"]] for p in pts]})

def demo_telemetry_loop():
    """Optional realtime demo: emits fake telemetry point moving in a circle."""
    import math
    center_lat, center_lon = 10.7769, 106.7009
    r = 0.002
    t = 0.0
    while True:
        lat = center_lat + r * math.cos(t)
        lon = center_lon + r * math.sin(t)
        socketio.emit("telemetry", {"lat": lat, "lon": lon})
        t += 0.12
        time.sleep(0.5)

if __name__ == "__main__":
    Thread(target=demo_telemetry_loop, daemon=True).start()
    socketio.run(app, host=APP_HOST, port=APP_PORT)
