"""
RescuOpt AI — Flask Backend Server (Clean)
============================================
รับผลจาก YOLO detection → relay ข้อมูลให้ dashboard frontend
Algorithms run in browser (disaster_nav.html), not here.
"""

try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
except ImportError as e:
    raise RuntimeError(
        "Missing dependency: install Flask and Flask-CORS with 'pip install flask flask-cors'"
    ) from e

import math, os, json
from pathlib import Path

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Global storage ──────────────────────────────────────
GLOBAL_HAZARDS = []
GLOBAL_SURVIVOR = None
GLOBAL_EXIT = None
EARTH_RADIUS_M = 6371000
DEFAULT_HAZARD_RADIUS_M = 500
DEFAULT_SAFETY_MARGIN_M = 500

# ── Coordinate system ──────────────────────────────────
REF_LAT = 13.7563
REF_LNG = 100.5018
DEG_M_LAT = 111320

def to_meters(lat, lng):
    dx = (lng - REF_LNG) * DEG_M_LAT * math.cos(math.radians(REF_LAT))
    dy = (lat - REF_LAT) * DEG_M_LAT
    return {"x": round(dx), "y": round(dy)}

def to_latlng(x, y):
    lng = REF_LNG + x / (DEG_M_LAT * math.cos(math.radians(REF_LAT)))
    lat = REF_LAT + y / DEG_M_LAT
    return [lat, lng]

def haversine(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def offset_coordinate(lat, lng, distance_m, bearing_deg):
    bearing = math.radians(bearing_deg)
    angular_distance = distance_m / EARTH_RADIUS_M
    lat1 = math.radians(lat)
    lng1 = math.radians(lng)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lng2 = lng1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return {"lat": math.degrees(lat2), "lng": math.degrees(lng2)}

def total_danger(mx, my, hazards):
    total = 0.0
    for h in hazards:
        hm = to_meters(h["lat"], h["lng"])
        dist = math.hypot(mx - hm["x"], my - hm["y"])
        radius = h.get("radius_m", 200)
        if dist < radius:
            total += (h.get("severity", 5) / 10.0) * (1.0 - dist / radius)
    return min(total, 1.0)

def exit_is_safe(exit_point, hazards, safety_margin_m=DEFAULT_SAFETY_MARGIN_M):
    return all(
        haversine(exit_point["lat"], exit_point["lng"], h["lat"], h["lng"])
        > (h.get("radius_m") or DEFAULT_HAZARD_RADIUS_M) + safety_margin_m
        for h in hazards
    )

def generate_safe_exit(survivor, hazards, safety_margin_m=DEFAULT_SAFETY_MARGIN_M):
    if not survivor:
        return None
    if not hazards:
        return offset_coordinate(survivor["lat"], survivor["lng"], 5000, 0)
    nearest = min(
        hazards,
        key=lambda h: haversine(survivor["lat"], survivor["lng"], h["lat"], h["lng"]),
    )
    nearest_distance = max(
        haversine(survivor["lat"], survivor["lng"], nearest["lat"], nearest["lng"]), 1.0,
    )
    dlat = math.radians(survivor["lat"] - nearest["lat"])
    dlng = math.radians(survivor["lng"] - nearest["lng"])
    y = math.sin(dlng) * math.cos(math.radians(survivor["lat"]))
    x = (
        math.cos(math.radians(nearest["lat"])) * math.sin(math.radians(survivor["lat"]))
        - math.sin(math.radians(nearest["lat"])) * math.cos(math.radians(survivor["lat"])) * math.cos(dlng)
    )
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360 if abs(dlat) + abs(dlng) > 1e-12 else 0
    minimum_distance = (nearest.get("radius_m") or DEFAULT_HAZARD_RADIUS_M) + safety_margin_m
    candidate_distance = max(nearest_distance + safety_margin_m, minimum_distance)
    for _ in range(60):
        candidate = offset_coordinate(nearest["lat"], nearest["lng"], candidate_distance, bearing)
        if exit_is_safe(candidate, hazards, safety_margin_m):
            return candidate
        candidate_distance += safety_margin_m
    return offset_coordinate(nearest["lat"], nearest["lng"], candidate_distance, bearing)

# ── Static file routes ────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

@app.route("/disaster_nav")
def disaster_nav_redirect():
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "dashboard"),
        "disaster_nav.html"
    )

@app.route("/dashboard/<path:filename>")
def serve_dashboard(filename):
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "dashboard"),
        filename
    )

@app.route("/api/hazards", methods=["GET"])
def get_hazards():
    return jsonify({
        "hazards": GLOBAL_HAZARDS,
        "survivor": GLOBAL_SURVIVOR,
        "exit": GLOBAL_EXIT,
    })

@app.route("/api/report_flood", methods=["POST"])
def report_flood():
    data = request.json
    severity = data.get("severity", 5)
    lat = data.get("lat", 13.7563)
    lng = data.get("lng", 100.5018)

    if severity <= 3:
        radius_m = 2000
    elif severity <= 6:
        radius_m = 20000
    else:
        radius_m = 100000

    new_hazard = {
        "type": "flood",
        "lat": lat, "lng": lng,
        "severity": severity, "radius_m": radius_m,
        "confidence": data.get("confidence", 0),
        "level_label": data.get("level_label", "Unknown")
    }
    GLOBAL_HAZARDS.append(new_hazard)

    user_dist_m = float(data.get("user_dist_m", 0))
    user_dir = data.get("user_dir", "N")
    bearings = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
    bearing_deg = bearings.get(user_dir, 0)

    total_dist_from_center_m = radius_m + user_dist_m
    survivor = offset_coordinate(lat, lng, total_dist_from_center_m, bearing_deg)

    global GLOBAL_SURVIVOR, GLOBAL_EXIT
    GLOBAL_SURVIVOR = survivor
    GLOBAL_EXIT = generate_safe_exit(survivor, GLOBAL_HAZARDS)

    return jsonify({"status": "received", "hazard": new_hazard})

@app.route("/api/report_flood_v2", methods=["POST"])
def report_flood_v2():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    survivors = data.get("survivors", [])
    hazards = data.get("hazards", [])
    severity = data.get("severity", 5)
    confidence = data.get("confidence", 0)
    level_label = data.get("level_label", "Unknown")

    GLOBAL_HAZARDS.clear()
    seen_hazards = set()

    for h in hazards:
        lat = h.get("lat", 13.7563)
        lng = h.get("lng", 100.5018)
        radius_m = h.get("radius_m") or DEFAULT_HAZARD_RADIUS_M
        hazard_key = (round(lat, 7), round(lng, 7), h.get("type", "flood"))
        if hazard_key in seen_hazards:
            continue
        seen_hazards.add(hazard_key)
        GLOBAL_HAZARDS.append({
            "type": h.get("type", "flood"),
            "lat": lat, "lng": lng,
            "severity": h.get("severity", severity),
            "radius_m": radius_m,
            "confidence": confidence,
            "level_label": level_label,
        })

    global GLOBAL_SURVIVOR, GLOBAL_EXIT
    if survivors:
        first = survivors[0]
        GLOBAL_SURVIVOR = {"lat": first["lat"], "lng": first["lng"]}
        GLOBAL_EXIT = generate_safe_exit(GLOBAL_SURVIVOR, GLOBAL_HAZARDS)

    return jsonify({
        "status": "received",
        "survivors_count": len(survivors),
        "hazards_count": len(GLOBAL_HAZARDS),
        "exit": GLOBAL_EXIT,
    })

if __name__ == "__main__":
    print("RescuOpt AI Server starting on http://127.0.0.1:5000/dashboard/disaster_nav.html")
    app.run(debug=True, port=5000)
