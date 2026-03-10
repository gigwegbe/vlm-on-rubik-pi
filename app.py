from flask import Flask, Response, send_from_directory, jsonify
from flask_cors import CORS
import os
import time
import base64
import json
import sqlite3
from datetime import datetime

import threading

vlm_lock = threading.Lock()

# VLM
from vlm_processor import initialize_vlm, process_image_for_gauges

app = Flask(__name__, static_folder="static")
CORS(app)

IMAGE_FOLDER = "merged_gauges_csv"
STREAM_INTERVAL = 30

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:9876")
VLM_REQUEST_TIMEOUT = int(os.getenv("VLM_REQUEST_TIMEOUT", "120"))

vlm_processor = None




# ------------------------------------------------
# Helpers
# ------------------------------------------------

def get_image_files():
    if not os.path.exists(IMAGE_FOLDER):
        return []

    exts = (".jpg", ".jpeg", ".png")
    return sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(exts)])


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ------------------------------------------------
# VLM
# ------------------------------------------------

# def process_image_with_vlm(image_path):

#     global vlm_processor

#     if vlm_processor is None:
#         print("Connecting to llama-server...")
#         vlm_processor = initialize_vlm(server_url=LLAMA_SERVER_URL)

#     result = process_image_for_gauges(image_path=image_path)
#     return result

def process_image_with_vlm(image_path):

    global vlm_processor

    try:

        with vlm_lock:  # prevents parallel requests

            if vlm_processor is None:
                print(f"Connecting to llama-server at {LLAMA_SERVER_URL}")
                vlm_processor = initialize_vlm(
                    server_url=LLAMA_SERVER_URL,
                    request_timeout=VLM_REQUEST_TIMEOUT
                )

            result = process_image_for_gauges(image_path=image_path)

            return result

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "gauge_readings": None
        }
# ------------------------------------------------
# Database
# ------------------------------------------------

def save_vlm_readings_to_db(vlm_result):

    if not vlm_result.get("success"):
        return

    readings = vlm_result.get("gauge_readings") or {}

    temperature = float(readings.get("thermometer") or 0)
    pressure = float(readings.get("pressure_gauge") or 0)
    rain = float(readings.get("rain_gauge") or 0)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect("sensors-json.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sensor_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        temperature REAL,
        pressure REAL,
        rain REAL
    )
    """)

    cur.execute(
        "INSERT INTO sensor_data (timestamp,temperature,pressure,rain) VALUES (?,?,?,?)",
        (timestamp, temperature, pressure, rain)
    )

    conn.commit()
    conn.close()


# ------------------------------------------------
# SSE STREAM
# ------------------------------------------------

def generate_stream():

    files = get_image_files()

    if not files:
        yield f"data: {json.dumps({'error':'no images'})}\n\n"
        return

    i = 0

    while True:

        img = files[i]
        path = os.path.join(IMAGE_FOLDER, img)

        try:

            encoded = encode_image(path)

            vlm_result = process_image_with_vlm(path)

            save_vlm_readings_to_db(vlm_result)

            payload = {
                "image": encoded,
                "filename": img,
                "timestamp": time.time(),
                "vlm": vlm_result
            }

            yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:

            yield f"data: {json.dumps({'error':str(e)})}\n\n"

        i = (i + 1) % len(files)

        time.sleep(STREAM_INTERVAL)


# ------------------------------------------------
# Routes
# ------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/stream")
def stream():
    return Response(
        generate_stream(),
        mimetype="text/event-stream"
    )


@app.route("/status")
def status():

    files = get_image_files()

    return jsonify({
        "status": "running",
        "images": len(files),
        "interval": STREAM_INTERVAL
    })


# ------------------------------------------------
# Run
# ------------------------------------------------

if __name__ == "__main__":

    print("Server starting")
    print("Images:", len(get_image_files()))
    print("Stream: http://localhost:5001/stream")

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )