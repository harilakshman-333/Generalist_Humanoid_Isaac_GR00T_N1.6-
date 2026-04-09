# vla_server.py
# Standalone FastAPI HTTP server exposing the VLA model as a REST API.
#
# This lets you test the VLA brain independently of ROS 2 and the full
# Docker stack — useful during development and CI.
#
# Endpoints:
#   GET  /health          — liveness check
#   POST /infer           — run one inference step
#   GET  /metrics         — latency and throughput stats
#
# Run:
#   pip install fastapi uvicorn pillow numpy
#   python3 vla_server.py --mock          # no model weights needed
#   python3 vla_server.py --model_id nvidia/GR00T-N1.6-3B
#
# Test:
#   curl -s http://localhost:8000/health | python3 -m json.tool
#   curl -s -X POST http://localhost:8000/infer \
#        -H "Content-Type: application/json" \
#        -d '{"instruction": "Pick up the red block", "image_b64": ""}' \
#        | python3 -m json.tool

from __future__ import annotations
import os
import sys
import time
import base64
import logging
import argparse
from typing import Optional

log = logging.getLogger("vla_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("[ERROR] FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class InferRequest(BaseModel):
    instruction: str = "Walk forward to the goal."
    image_b64: str = ""          # Base64-encoded JPEG/PNG image (empty = use mock frame)
    session_id: str = "default"


class InferResponse(BaseModel):
    action_type: str
    target_xyz: list[float]
    target_rpy: list[float]
    gripper_open: bool
    confidence: float
    latency_ms: float
    session_id: str


class HealthResponse(BaseModel):
    status: str
    model_id: str
    mock_mode: bool
    uptime_s: float


class MetricsResponse(BaseModel):
    total_requests: int
    mean_latency_ms: float
    max_latency_ms: float
    requests_per_second: float


# ---------------------------------------------------------------------------
# App State
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.model = None
        self.mock = True
        self.model_id = "nvidia/GR00T-N1.6-3B"
        self.start_time = time.time()
        self.request_count = 0
        self.latencies: list[float] = []

    def record_latency(self, ms: float):
        self.latencies.append(ms)
        if len(self.latencies) > 1000:
            self.latencies.pop(0)

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def max_latency_ms(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def rps(self) -> float:
        uptime = time.time() - self.start_time
        return self.request_count / uptime if uptime > 0 else 0.0


# ---------------------------------------------------------------------------
# Build app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GR00T VLA Inference Server",
    description="REST API for Unitree G1 Vision-Language-Action inference",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = AppState()


@app.on_event("startup")
async def load_model():
    """Load the VLA model at server startup."""
    # Import here to avoid circular deps with run_inference.py
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from run_inference import VLAModel
        log.info(f"Loading VLA model (mock={state.mock})...")
        state.model = VLAModel(state.model_id, device="cuda", mock=state.mock)
        log.info("Model ready.")
    except Exception as e:
        log.error(f"Model load failed: {e}")
        # Keep server alive in mock mode
        state.mock = True
        from run_inference import VLAModel
        state.model = VLAModel(state.model_id, device="cpu", mock=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_id=state.model_id,
        mock_mode=state.mock,
        uptime_s=round(time.time() - state.start_time, 1),
    )


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")

    # Decode image
    frame = None
    if req.image_b64:
        try:
            import numpy as np
            from PIL import Image as PILImage
            import io
            img_bytes = base64.b64decode(req.image_b64)
            img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            frame = np.array(img)
        except Exception as e:
            log.warning(f"Image decode failed: {e} — using mock frame.")

    if frame is None:
        sys.path.insert(0, os.path.dirname(__file__))
        from run_inference import get_latest_frame
        frame = get_latest_frame(mock=True)

    # Run inference
    t0 = time.perf_counter()
    action = state.model.infer(frame, req.instruction)
    latency_ms = (time.perf_counter() - t0) * 1000

    state.request_count += 1
    state.record_latency(latency_ms)

    log.info(
        f"[{req.session_id}] {action['action_type']} | "
        f"xyz={action['target_xyz']} | {latency_ms:.1f}ms"
    )

    return InferResponse(
        action_type=action["action_type"],
        target_xyz=action["target_xyz"],
        target_rpy=action["target_rpy"],
        gripper_open=action["gripper_open"],
        confidence=action["confidence"],
        latency_ms=round(latency_ms, 2),
        session_id=req.session_id,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return MetricsResponse(
        total_requests=state.request_count,
        mean_latency_ms=round(state.mean_latency_ms, 2),
        max_latency_ms=round(state.max_latency_ms, 2),
        requests_per_second=round(state.rps, 3),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host",     default="0.0.0.0")
    p.add_argument("--port",     type=int, default=8000)
    p.add_argument("--model_id", default="nvidia/GR00T-N1.6-3B")
    p.add_argument("--mock",     action="store_true", default=True,
                   help="Run without real model weights (for testing)")
    p.add_argument("--real",     action="store_true",
                   help="Load real model weights (disables mock)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    state.model_id = args.model_id
    state.mock = not args.real  # --real flag disables mock

    log.info(f"Starting VLA server on {args.host}:{args.port} | mock={state.mock}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
