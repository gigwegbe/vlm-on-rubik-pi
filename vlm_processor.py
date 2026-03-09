"""
VLM Processor Module for Gauge Reading Extraction
Calls a running llama-server instance via its OpenAI-compatible REST API.

Start your server with something like:
  llama-server \
    -m ./LFM2-VL-450M-Q8_0.gguf \
    --mmproj ./mmproj-LFM2-VL-450M-Q8_0.gguf \
    --no-mmproj-offload \
    -ngl 12 -b 24 -c 1536 \
    --host 0.0.0.0 --port 9876


    
  llama-server \
    -m ./LFM2-VL-450M-Q8_0.gguf \
    --mmproj ./mmproj-LFM2-VL-450M-Q8_0.gguf \
    --no-mmproj-offload \
    -ngl 12 -b 24 -c 2048 \
    --host 0.0.0.0 --port 9876
"""

import base64
import io
import json
import logging
import os
import traceback

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
GAUGE_PROMPT = """TASK: Extract numeric readings from three digital gauges in this image.

GAUGE IDENTIFICATION (left to right):
- LEFT gauge (black/dark): rain_gauge (units: mm)
- MIDDLE gauge (white with blue header): thermometer (units: °C)
- RIGHT gauge (white/red circular): pressure_gauge (units: bar)

READING INSTRUCTIONS:
1. Focus ONLY on the main numeric display on each gauge's LCD/LED screen
2. Read the complete number including decimal points if present
3. Ignore any secondary displays, unit labels, or interface elements
4. If a gauge shows multiple numbers, use the largest/primary display

OUTPUT FORMAT:
- Return ONLY valid JSON with no additional text, markdown, or formatting
- Use null for unreadable or missing gauges
- Round to maximum 2 decimal places
- Use integers when the value is a whole number

REQUIRED JSON STRUCTURE:
{
 "rain_gauge": <number|null>,
 "thermometer": <number|null>,
 "pressure_gauge": <number|null>
}

Analyze the image now and return the JSON response."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG data URL."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# VLMProcessor
# ---------------------------------------------------------------------------

class VLMProcessor:
    """
    Sends vision inference requests to a running llama-server instance.
    No Python llama.cpp bindings required — pure HTTP.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:9876",
        max_tokens: int = 512,
        temperature: float = 0.0,
        request_timeout: int = 120,
    ):
        """
        Args:
            server_url:      Base URL of the llama-server (no trailing slash).
            max_tokens:      Maximum tokens to generate.
            temperature:     Sampling temperature (0 = deterministic).
            request_timeout: HTTP request timeout in seconds.
        """
        self.server_url      = server_url.rstrip("/")
        self.chat_endpoint   = f"{self.server_url}/v1/chat/completions"
        self.health_endpoint = f"{self.server_url}/health"
        self.max_tokens      = max_tokens
        self.temperature     = temperature
        self.request_timeout = request_timeout
        self.is_initialized  = False

    # ------------------------------------------------------------------
    # Initialisation  (health-check ping only — model already loaded)
    # ------------------------------------------------------------------

    def initialize_models(self):
        """
        Verify the llama-server is reachable and ready.
        No models to load locally — they are already served by llama-server.
        """
        try:
            logger.info(f"Connecting to llama-server at {self.server_url} ...")
            resp = requests.get(self.health_endpoint, timeout=10)
            resp.raise_for_status()
            health = resp.json()
            status = health.get("status", "unknown")
            logger.info(f"llama-server health: {status}")

            if status not in ("ok", "no slot available"):
                raise RuntimeError(f"Unexpected health status from server: {status}")

            self.is_initialized = True
            logger.info("llama-server connection verified successfully!")

        except requests.exceptions.ConnectionError:
            msg = (
                f"Cannot reach llama-server at {self.server_url}. "
                "Make sure the server is running before starting this app."
            )
            logger.error(msg)
            self.is_initialized = False
            raise RuntimeError(msg)

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.is_initialized = False
            raise

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process_image(self, image_path: str = None, pil_image: Image.Image = None) -> dict:
        """
        Send an image to the llama-server and extract gauge readings.

        Args:
            image_path: Path to an image file.
            pil_image:  PIL Image object (takes priority over image_path).

        Returns:
            dict: { success, error, gauge_readings, raw_response }
        """
        if not self.is_initialized:
            return self._error("VLM processor not initialised. Call initialize_models() first.")

        try:
            # ---- load image ------------------------------------------------
            if pil_image is not None:
                image = pil_image
            elif image_path is not None:
                image = Image.open(image_path)
            else:
                return self._error("No image provided (pass image_path or pil_image).")

            if image.mode != "RGB":
                image = image.convert("RGB")

            data_url = pil_to_data_url(image)

            # ---- build OpenAI-compatible chat payload ----------------------
            payload = {
                "model": "local-model",   # llama-server ignores this field
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                            {
                                "type": "text",
                                "text": GAUGE_PROMPT,
                            },
                        ],
                    }
                ],
            }

            # ---- POST to llama-server --------------------------------------
            logger.info(f"Sending image to {self.chat_endpoint} ...")
            resp = requests.post(
                self.chat_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )
            resp.raise_for_status()

            raw_text: str = resp.json()["choices"][0]["message"]["content"]
            logger.info(f"VLM raw response: {raw_text}")

            gauge_readings = self._parse_gauge_response(raw_text)

            return {
                "success":        True,
                "error":          None,
                "gauge_readings": gauge_readings,
                "raw_response":   raw_text,
            }

        except requests.exceptions.Timeout:
            return self._error(
                f"Request timed out after {self.request_timeout}s. "
                "Try increasing request_timeout or reducing image size."
            )
        except requests.exceptions.RequestException as e:
            return self._error(f"HTTP error communicating with llama-server: {e}")
        except Exception as e:
            logger.error(traceback.format_exc())
            return self._error(f"Unexpected error processing image: {e}")

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_gauge_response(self, response: str) -> dict | None:
        """Extract JSON gauge readings from the raw model response."""
        try:
            response = response.strip()
            start = response.find("{")
            end   = response.rfind("}")

            if start == -1 or end == -1:
                logger.warning(f"No JSON object found in response: {response!r}")
                return None

            data = json.loads(response[start: end + 1])

            missing = {"rain_gauge", "thermometer", "pressure_gauge"} - data.keys()
            if missing:
                logger.warning(f"Response missing expected keys {missing}: {data}")

            return data   # partial data is still useful

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e} | raw: {response!r}")
            return None
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error(message: str) -> dict:
        logger.error(message)
        return {
            "success":        False,
            "error":          message,
            "gauge_readings": None,
            "raw_response":   None,
        }


# ---------------------------------------------------------------------------
# Module-level singleton  (same public API as before)
# ---------------------------------------------------------------------------

_vlm_processor: VLMProcessor | None = None


def get_vlm_processor(server_url: str = "http://localhost:9876", **kwargs) -> VLMProcessor:
    """Return the global VLMProcessor, creating it if necessary."""
    global _vlm_processor
    if _vlm_processor is None:
        _vlm_processor = VLMProcessor(server_url=server_url, **kwargs)
    return _vlm_processor


def initialize_vlm(server_url: str = "http://localhost:9876", **kwargs) -> VLMProcessor:
    """Ping the llama-server and return the ready processor."""
    processor = get_vlm_processor(server_url=server_url, **kwargs)
    if not processor.is_initialized:
        processor.initialize_models()
    return processor


def process_image_for_gauges(image_path: str = None, pil_image: Image.Image = None) -> dict:
    """Convenience wrapper — processor must already be initialised."""
    return get_vlm_processor().process_image(image_path=image_path, pil_image=pil_image)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:9876")
    TEST_IMAGE = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "merged_gauges_csv/merged_0001_caliper_2.27mm_temperature_27.2C_pressure_0.97bar.jpg"
    )

    print(f"Testing VLM Processor -> {SERVER_URL}")
    proc = initialize_vlm(server_url=SERVER_URL)

    if os.path.exists(TEST_IMAGE):
        result = proc.process_image(image_path=TEST_IMAGE)
        print(json.dumps(result, indent=2))
    else:
        print(f"Test image not found: {TEST_IMAGE}")
        print("Usage: python vlm_processor.py <path/to/image.jpg>")
