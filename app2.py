from flask import Flask, Response, render_template_string
from flask_cors import CORS
import os
import time
import base64
import json
import sqlite3
from datetime import datetime

# ---------------------------------------------------------------------------
# VLM processor (llama-server HTTP backend)
# ---------------------------------------------------------------------------
try:
    from vlm_processor import initialize_vlm, process_image_for_gauges
    VLM_AVAILABLE = True
    print("VLM processor imported successfully!")
except ImportError as e:
    print(f"Warning: VLM processor not available: {e}")
    VLM_AVAILABLE = False


app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_FOLDER = "merged_gauges_csv"
STREAM_INTERVAL = 10

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:9876")
VLM_REQUEST_TIMEOUT = int(os.getenv("VLM_REQUEST_TIMEOUT", "120"))

ENABLE_VLM = VLM_AVAILABLE

vlm_processor = None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def get_image_files():
    if not os.path.exists(IMAGE_FOLDER):
        return []

    exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

    return sorted(
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(exts)
    )


def encode_image_to_base64(image_path):

    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    except Exception as e:
        print("Error encoding image:", e)
        return None


# ---------------------------------------------------------------------------
# VLM processing
# ---------------------------------------------------------------------------

def process_image_with_vlm(image_path):

    global vlm_processor

    if not ENABLE_VLM or not VLM_AVAILABLE:

        return {
            "success": False,
            "error": "VLM disabled",
            "gauge_readings": None,
            "processing_time": 0
        }

    try:

        start_time = time.time()

        if vlm_processor is None:

            print("Connecting to llama-server...")

            vlm_processor = initialize_vlm(
                server_url=LLAMA_SERVER_URL,
                request_timeout=VLM_REQUEST_TIMEOUT
            )

            print("Connected to llama-server")

        result = process_image_for_gauges(image_path=image_path)

        result["processing_time"] = round(time.time() - start_time, 2)

        return result

    except Exception as e:

        print("VLM error:", e)

        return {
            "success": False,
            "error": str(e),
            "gauge_readings": None,
            "processing_time": 0
        }


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def save_vlm_readings_to_db(vlm_result):

    if not vlm_result.get("success"):
        return

    readings = vlm_result.get("gauge_readings", {})

    temperature = float(readings.get("thermometer", 0) or 0)
    pressure = float(readings.get("pressure_gauge", 0) or 0)
    rain = float(readings.get("rain_gauge", 0) or 0)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect("sensors-json.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sensor_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        temperature REAL,
        pressure REAL,
        rain REAL
    )
    """)

    cur.execute(
        "INSERT INTO sensor_data(timestamp,temperature,pressure,rain) VALUES(?,?,?,?)",
        (timestamp, temperature, pressure, rain)
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------

def generate_image_stream():

    image_files = get_image_files()

    if not image_files:
        yield f"data: {json.dumps({'error':'No images found'})}\n\n"
        return

    image_index = 0

    while True:

        try:

            current_image = image_files[image_index]

            image_path = os.path.join(IMAGE_FOLDER, current_image)

            encoded_image = encode_image_to_base64(image_path)

            if encoded_image:

                # SEND IMAGE FIRST
                image_event = {
                    "image": encoded_image,
                    "filename": current_image,
                    "index": image_index + 1,
                    "total": len(image_files),
                    "timestamp": time.time(),
                    "vlm_analysis": None
                }

                yield f"data: {json.dumps(image_event)}\n\n"

                # PROCESS VLM AFTER
                vlm_result = process_image_with_vlm(image_path)

                save_vlm_readings_to_db(vlm_result)

                vlm_event = {
                    "filename": current_image,
                    "timestamp": time.time(),
                    "vlm_analysis": vlm_result
                }

                yield f"data: {json.dumps(vlm_event)}\n\n"

            else:

                yield f"data: {json.dumps({'error':'Image load error'})}\n\n"

            image_index = (image_index + 1) % len(image_files)

            time.sleep(STREAM_INTERVAL)

        except Exception as e:

            yield f"data: {json.dumps({'error':str(e)})}\n\n"
            time.sleep(STREAM_INTERVAL)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(CLIENT_HTML)


@app.route("/stream")
def stream():

    return Response(
        generate_image_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.route("/status")
def status():

    image_files = get_image_files()

    return {
        "status": "running",
        "images": len(image_files),
        "interval": STREAM_INTERVAL,
        "vlm_enabled": ENABLE_VLM
    }


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

CLIENT_HTML = """
<!DOCTYPE html>
<html>

<head>

<title>Rubik-Pi Vision Dashboard</title>

<style>

body{
font-family:Arial;
background:#f4f6f9;
margin:20px;
}

.container{
max-width:1000px;
margin:auto;
}

img{
max-width:100%;
border-radius:8px;
}

.gauge{
margin-top:10px;
font-size:20px;
}

</style>

</head>

<body>

<div class="container">

<h2>Rubik-Pi Vision Dashboard</h2>

<p id="status">Connecting...</p>

<img id="img">

<div id="gauges"></div>

</div>

<script>

const img=document.getElementById("img")
const gauges=document.getElementById("gauges")
const status=document.getElementById("status")

const es=new EventSource("/stream")

es.onopen=()=>{
status.innerText="Connected"
}

es.onmessage=(event)=>{

let data=JSON.parse(event.data)

if(data.image){

img.src="data:image/jpeg;base64,"+data.image

}

if(data.vlm_analysis){

let r=data.vlm_analysis.gauge_readings

if(!r)return

gauges.innerHTML=`

<div class='gauge'>🌡 Temp: ${r.thermometer ?? "-"} °C</div>
<div class='gauge'>⚙ Pressure: ${r.pressure_gauge ?? "-"} bar</div>
<div class='gauge'>🌧 Rain: ${r.rain_gauge ?? "-"} mm</div>

`

}

}

</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("Starting Flask Vision Dashboard")

    print("Image folder:", IMAGE_FOLDER)
    print("Stream interval:", STREAM_INTERVAL)
    print("llama-server:", LLAMA_SERVER_URL)

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )