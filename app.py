from flask import Flask, Response, render_template_string
from flask_cors import CORS
import os
import time
import base64
import json
from PIL import Image
import io
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
IMAGE_FOLDER    = "merged_gauges_csv"
STREAM_INTERVAL = 10   # seconds between frames

# llama-server connection — override with env vars if needed
LLAMA_SERVER_URL    = os.getenv("LLAMA_SERVER_URL",    "http://localhost:9876")
VLM_REQUEST_TIMEOUT = int(os.getenv("VLM_REQUEST_TIMEOUT", "120"))   # seconds

ENABLE_VLM = VLM_AVAILABLE

# Lazy-initialised on first request
vlm_processor = None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def get_image_files():
    if not os.path.exists(IMAGE_FOLDER):
        return []
    exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    return sorted(f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(exts))


def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# VLM processing
# ---------------------------------------------------------------------------

def process_image_with_vlm(image_path):
    """Forward an image to the llama-server and return gauge readings."""
    global vlm_processor

    if not ENABLE_VLM or not VLM_AVAILABLE:
        return {
            "success":         False,
            "error":           "VLM processing disabled or not available",
            "gauge_readings":  None,
            "processing_time": 0,
        }

    try:
        start_time = time.time()

        # Lazy init — just a health-check ping, model is already loaded server-side
        if vlm_processor is None:
            print(f"Connecting to llama-server at {LLAMA_SERVER_URL} ...")
            vlm_processor = initialize_vlm(
                server_url=LLAMA_SERVER_URL,
                request_timeout=VLM_REQUEST_TIMEOUT,
            )
            print("llama-server connection established!")

        result = process_image_for_gauges(image_path=image_path)
        result["processing_time"] = round(time.time() - start_time, 2)
        return result

    except Exception as e:
        error_msg = f"VLM processing error: {e}"
        print(error_msg)
        return {
            "success":         False,
            "error":           error_msg,
            "gauge_readings":  None,
            "processing_time": 0,
        }


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def save_vlm_readings_to_db(vlm_result):
    """Persist gauge readings to SQLite."""
    if not vlm_result.get("success"):
        return

    readings    = vlm_result.get("gauge_readings", {}) or {}
    temperature = float(readings.get("thermometer",    0) or 0)
    pressure    = float(readings.get("pressure_gauge", 0) or 0)
    rain        = float(readings.get("rain_gauge",     0) or 0)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect("sensors-json.db")
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            temperature REAL,
            pressure    REAL,
            rain        REAL
        )
    """)
    cur.execute(
        "INSERT INTO sensor_data (timestamp, temperature, pressure, rain) VALUES (?, ?, ?, ?)",
        (timestamp, temperature, pressure, rain),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# SSE stream generator
# ---------------------------------------------------------------------------

def generate_image_stream():
    image_files = get_image_files()

    if not image_files:
        yield f"data: {json.dumps({'error': 'No images found in folder'})}\n\n"
        return

    image_index = 0

    while True:
        try:
            current_image = image_files[image_index]
            image_path    = os.path.join(IMAGE_FOLDER, current_image)

            encoded_image = encode_image_to_base64(image_path)

            if encoded_image:
                vlm_result = process_image_with_vlm(image_path)
                save_vlm_readings_to_db(vlm_result)

                data = {
                    "image":        encoded_image,
                    "filename":     current_image,
                    "index":        image_index + 1,
                    "total":        len(image_files),
                    "timestamp":    time.time(),
                    "vlm_analysis": vlm_result,
                }
                yield f"data: {json.dumps(data)}\n\n"

                if vlm_result["success"]:
                    print(
                        f"[{image_index+1}/{len(image_files)}] {current_image} | "
                        f"readings={vlm_result.get('gauge_readings')} | "
                        f"time={vlm_result.get('processing_time', 0)}s"
                    )
                else:
                    print(
                        f"[{image_index+1}/{len(image_files)}] {current_image} | "
                        f"VLM error: {vlm_result.get('error', 'unknown')}"
                    )
            else:
                yield f"data: {json.dumps({'error': f'Could not load image: {current_image}', 'timestamp': time.time()})}\n\n"

            image_index = (image_index + 1) % len(image_files)
            time.sleep(STREAM_INTERVAL)

        except Exception as e:
            yield f"data: {json.dumps({'error': f'Stream error: {e}', 'timestamp': time.time()})}\n\n"
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
            "Cache-Control":               "no-cache",
            "Connection":                  "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@app.route("/status")
def status():
    image_files = get_image_files()
    return {
        "status":          "running",
        "image_folder":    IMAGE_FOLDER,
        "stream_interval": STREAM_INTERVAL,
        "total_images":    len(image_files),
        "image_files":     image_files[:10],
        "vlm_config": {
            "backend":         "llama-server (HTTP)",
            "server_url":      LLAMA_SERVER_URL,
            "request_timeout": VLM_REQUEST_TIMEOUT,
            "available":       VLM_AVAILABLE,
            "enabled":         ENABLE_VLM,
            "connected":       vlm_processor is not None,
        },
    }


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

# CLIENT_HTML = '''
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Image Stream Client</title>
#     <style>
#         body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
#         .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
#         .status { background-color: #e8f5e8; padding: 10px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #4CAF50; }
#         .error { background-color: #ffe8e8; padding: 10px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #f44336; }
#         .image-container { text-align: center; margin: 20px 0; }
#         .image-container img { max-width: 100%; max-height: 600px; border: 2px solid #ddd; border-radius: 5px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
#         .image-info { margin-top: 10px; font-size: 14px; color: #666; }
#         .vlm-analysis { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 15px; margin-top: 20px; }
#         .gauge-readings { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 10px; }
#         .gauge-item { background-color: white; padding: 10px; border-radius: 3px; border-left: 4px solid #007bff; }
#         .gauge-label { font-weight: bold; color: #495057; }
#         .gauge-value { font-size: 18px; color: #007bff; margin-top: 5px; }
#         .vlm-error { color: #dc3545; font-style: italic; }
#         .processing-time { font-size: 12px; color: #6c757d; margin-top: 5px; }
#         .loading { text-align: center; padding: 50px; font-size: 18px; color: #666; }
#         .connection-status { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
#         .connected    { background-color: #4CAF50; }
#         .disconnected { background-color: #f44336; }
#         .connecting   { background-color: #ff9800; }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>Image Stream Client</h1>
#         <div class="status">
#             <span class="connection-status" id="connectionStatus"></span>
#             <span id="statusText">Connecting to stream...</span>
#         </div>
#         <div id="errorContainer"></div>
#         <div class="loading" id="loadingMessage">Waiting for images from server...</div>
#         <div class="image-container" id="imageContainer" style="display: none;">
#             <img id="streamedImage" src="" alt="Streamed Image">
#             <div class="image-info" id="imageInfo"></div>
#             <div class="vlm-analysis" id="vlmAnalysis">
#                 <h3>🔍 Gauge Analysis from Vision Language Model:</h3>
#                 <div id="vlmContent"><p>Processing image with VLM...</p></div>
#             </div>
#         </div>
#     </div>

#     <script>
#         const statusElement    = document.getElementById('statusText');
#         const connectionStatus = document.getElementById('connectionStatus');
#         const errorContainer   = document.getElementById('errorContainer');
#         const loadingMessage   = document.getElementById('loadingMessage');
#         const imageContainer   = document.getElementById('imageContainer');
#         const streamedImage    = document.getElementById('streamedImage');
#         const imageInfo        = document.getElementById('imageInfo');
#         const vlmContent       = document.getElementById('vlmContent');
#         let eventSource, reconnectTimeout;

#         function updateConnectionStatus(status, message) {
#             statusElement.textContent = message;
#             connectionStatus.className = 'connection-status ' + status;
#         }
#         function showError(msg) { errorContainer.innerHTML = `<div class="error">${msg}</div>`; }
#         function clearError()   { errorContainer.innerHTML = ''; }

#         function displayVLMAnalysis(vlmData) {
#             if (!vlmData) { vlmContent.innerHTML = '<p class="vlm-error">No VLM analysis data</p>'; return; }
#             if (!vlmData.success) { vlmContent.innerHTML = `<p class="vlm-error">VLM Error: ${vlmData.error || 'Unknown'}</p>`; return; }
#             const readings = vlmData.gauge_readings;
#             const pt       = vlmData.processing_time || 0;
#             if (!readings) { vlmContent.innerHTML = '<p class="vlm-error">No gauge readings extracted</p>'; return; }

#             const gaugeConfig = [
#                 { key: 'rain_gauge',     label: 'Rain Gauge',  unit: 'mm',  color: '#28a745' },
#                 { key: 'thermometer',    label: 'Temperature', unit: '°C',  color: '#dc3545' },
#                 { key: 'pressure_gauge', label: 'Pressure',    unit: 'bar', color: '#007bff' },
#             ];

#             let html = '<div class="gauge-readings">';
#             gaugeConfig.forEach(g => {
#                 const v = readings[g.key];
#                 const display = (v !== null && v !== undefined) ? `${v} ${g.unit}` : 'Not detected';
#                 html += `<div class="gauge-item" style="border-left-color:${g.color}">
#                             <div class="gauge-label">${g.label}</div>
#                             <div class="gauge-value" style="color:${g.color}">${display}</div>
#                          </div>`;
#             });
#             html += `</div><div class="processing-time">Processed in ${pt}s</div>`;
#             vlmContent.innerHTML = html;
#         }

#         function connectToStream() {
#             if (eventSource) eventSource.close();
#             updateConnectionStatus('connecting', 'Connecting to stream...');
#             eventSource = new EventSource('/stream');

#             eventSource.onopen = () => {
#                 updateConnectionStatus('connected', 'Connected – streaming every 60 s');
#                 clearError();
#             };

#             eventSource.onmessage = (event) => {
#                 try {
#                     const data = JSON.parse(event.data);
#                     if (data.error) { showError(`Server Error: ${data.error}`); return; }
#                     if (data.image) {
#                         loadingMessage.style.display = 'none';
#                         imageContainer.style.display = 'block';
#                         streamedImage.src = `data:image/jpeg;base64,${data.image}`;
#                         const ts = new Date(data.timestamp * 1000).toLocaleTimeString();
#                         imageInfo.innerHTML = `<strong>Time:</strong> ${ts}`;
#                         displayVLMAnalysis(data.vlm_analysis);
#                     }
#                 } catch (err) { showError(`Error parsing data: ${err.message}`); }
#             };

#             eventSource.onerror = () => {
#                 updateConnectionStatus('disconnected', 'Connection lost – reconnecting...');
#                 clearTimeout(reconnectTimeout);
#                 reconnectTimeout = setTimeout(connectToStream, 5000);
#             };
#         }

#         window.addEventListener('load', connectToStream);
#         window.addEventListener('beforeunload', () => {
#             if (eventSource) eventSource.close();
#             clearTimeout(reconnectTimeout);
#         });
#     </script>
# </body>
# </html>
# '''

CLIENT_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Rubik-Pi Vision Dashboard</title>

<style>

body{
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial;
    margin:0;
    background:#f4f6f9;
}

.container{
    max-width:1200px;
    margin:auto;
    padding:20px;
}

header{
    display:flex;
    align-items:center;
    gap:15px;
    padding-bottom:15px;
    border-bottom:1px solid #ddd;
    margin-bottom:20px;
}

.logo{
    width:60px;
}

.title-group h1{
    margin:0;
    font-size:28px;
}

.subtitle{
    margin:0;
    color:#666;
    font-size:14px;
}

.status{
    background:#e8f5e8;
    padding:12px;
    border-radius:6px;
    margin-bottom:20px;
    border-left:5px solid #4CAF50;
    font-size:14px;
}

.connection-status{
    display:inline-block;
    width:10px;
    height:10px;
    border-radius:50%;
    margin-right:6px;
}

.connected{background:#4CAF50;}
.disconnected{background:#f44336;}
.connecting{background:#ff9800;}

.image-container{
    text-align:center;
    background:white;
    padding:15px;
    border-radius:10px;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

.image-container img{
    max-width:100%;
    max-height:550px;
    border-radius:6px;
}

.image-info{
    margin-top:10px;
    color:#666;
    font-size:14px;
}

.vlm-analysis{
    margin-top:25px;
    background:white;
    padding:20px;
    border-radius:10px;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

.vlm-analysis h3{
    margin-top:0;
}

.gauge-readings{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    gap:15px;
    margin-top:15px;
}

.gauge-item{
    padding:15px;
    border-radius:6px;
    background:#f9fafb;
    border-left:5px solid #007bff;
}

.gauge-label{
    font-weight:600;
    color:#444;
}

.gauge-value{
    font-size:22px;
    margin-top:5px;
}

.processing-time{
    margin-top:12px;
    font-size:12px;
    color:#777;
}

.loading{
    text-align:center;
    padding:40px;
    color:#777;
    font-size:18px;
}

.error{
    background:#ffe8e8;
    padding:10px;
    border-radius:6px;
    border-left:5px solid #f44336;
    margin-bottom:20px;
}

</style>
</head>

<body>

<div class="container">

<header>
<img src="/static/images.png" class="logo">

<div class="title-group">
<h1>Rubik-Pi Vision Dashboard</h1>
<p class="subtitle">Real-time Gauge Monitoring with On-Device Vision AI</p>
</div>

</header>

<div class="status">
<span class="connection-status" id="connectionStatus"></span>
<span id="statusText">Connecting to stream...</span>
</div>

<div id="errorContainer"></div>

<div class="loading" id="loadingMessage">
Waiting for images from server...
</div>

<div class="image-container" id="imageContainer" style="display:none">

<img id="streamedImage" src="" alt="Streamed Image">

<div class="image-info" id="imageInfo"></div>

</div>

<div class="vlm-analysis" id="vlmAnalysis">

<h3>🔎 Gauge Analysis from Vision Language Model:</h3>

<div id="vlmContent">
<p>Processing image with VLM...</p>
</div>

</div>

</div>

<script>

const statusElement=document.getElementById('statusText')
const connectionStatus=document.getElementById('connectionStatus')
const errorContainer=document.getElementById('errorContainer')
const loadingMessage=document.getElementById('loadingMessage')
const imageContainer=document.getElementById('imageContainer')
const streamedImage=document.getElementById('streamedImage')
const imageInfo=document.getElementById('imageInfo')
const vlmContent=document.getElementById('vlmContent')

let eventSource
let reconnectTimeout

function updateConnectionStatus(status,message){
statusElement.textContent=message
connectionStatus.className='connection-status '+status
}

function showError(msg){
errorContainer.innerHTML=`<div class="error">${msg}</div>`
}

function clearError(){
errorContainer.innerHTML=''
}

function displayVLMAnalysis(vlmData){

if(!vlmData){
vlmContent.innerHTML='<p>No VLM analysis</p>'
return
}

if(!vlmData.success){
vlmContent.innerHTML=`<p>VLM Error: ${vlmData.error}</p>`
return
}

const readings=vlmData.gauge_readings
const pt=vlmData.processing_time||0

const config=[
{key:'rain_gauge',label:'Rain Gauge',unit:'mm',color:'#28a745'},
{key:'thermometer',label:'Temperature',unit:'°C',color:'#dc3545'},
{key:'pressure_gauge',label:'Pressure',unit:'bar',color:'#007bff'}
]

let html='<div class="gauge-readings">'

config.forEach(g=>{
const v=readings[g.key]
const val=v!==undefined&&v!==null?`${v} ${g.unit}`:'Not detected'

html+=`
<div class="gauge-item" style="border-left-color:${g.color}">
<div class="gauge-label">${g.label}</div>
<div class="gauge-value" style="color:${g.color}">
${val}
</div>
</div>`
})

html+='</div>'
html+=`<div class="processing-time">Processed in ${pt}s</div>`

vlmContent.innerHTML=html

}

function connectToStream(){

if(eventSource)eventSource.close()

updateConnectionStatus('connecting','Connecting to stream...')

eventSource=new EventSource('/stream')

eventSource.onopen=()=>{
updateConnectionStatus('connected','Connected – streaming every 60s')
clearError()
}

eventSource.onmessage=(event)=>{

try{

const data=JSON.parse(event.data)

if(data.error){
showError(data.error)
return
}

if(data.image){

loadingMessage.style.display='none'
imageContainer.style.display='block'

streamedImage.src=`data:image/jpeg;base64,${data.image}`

const ts=new Date(data.timestamp*1000).toLocaleTimeString()

imageInfo.innerHTML=`<strong>Time:</strong> ${ts}`

displayVLMAnalysis(data.vlm_analysis)

}

}catch(err){

showError(err.message)

}

}

eventSource.onerror=()=>{
updateConnectionStatus('disconnected','Connection lost – reconnecting...')
clearTimeout(reconnectTimeout)
reconnectTimeout=setTimeout(connectToStream,5000)
}

}

window.addEventListener('load',connectToStream)

window.addEventListener('beforeunload',()=>{
if(eventSource)eventSource.close()
clearTimeout(reconnectTimeout)
})

</script>

</body>
</html>
'''

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Flask Image Streaming Server with llama-server VLM integration...")
    print(f"  Image folder     : {IMAGE_FOLDER}")
    print(f"  Stream interval  : {STREAM_INTERVAL}s")
    print(f"  llama-server URL : {LLAMA_SERVER_URL}")
    print(f"  Request timeout  : {VLM_REQUEST_TIMEOUT}s")
    print(f"  VLM available    : {VLM_AVAILABLE}")
    print(f"  VLM enabled      : {ENABLE_VLM}")

    image_files = get_image_files()
    print(f"  Images found     : {len(image_files)}")
    if image_files:
        print("  Sample images    :", image_files[:5])
    else:
        print("  Warning: No images found in the specified folder!")

    print("\nServer endpoints:")
    print("  Client  : http://localhost:5001/")
    print("  Stream  : http://localhost:5001/stream")
    print("  Status  : http://localhost:5001/status")

    if VLM_AVAILABLE and ENABLE_VLM:
        print(f"\n⚡ llama-server VLM active at {LLAMA_SERVER_URL} — images will be analysed for gauge readings!")
    else:
        print("\n⚠️  VLM integration disabled — only image streaming will be active.")

    app.run(debug=True, host="0.0.0.0", port=5001)