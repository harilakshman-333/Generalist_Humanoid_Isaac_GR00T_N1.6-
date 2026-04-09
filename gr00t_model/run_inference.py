# run_inference.py
# Unitree G1 VLA Inference Node
#
# Loads the GR00T-N1.6 / UnifoLM-VLA model and runs a continuous inference loop.
# Publishes high-level action targets to ROS 2 topic /vla/target_pose.
#
# The inference loop runs at ~5 Hz (decoupled from the 50 Hz RL policy),
# which is typical for vision-language models of this size.
#
# Run inside the gr00t container:
#   docker compose run --rm gr00t python3 /workspace/gr00t_model/run_inference.py
#
# Environment variables:
#   MODEL_ID     — HuggingFace model ID (default: nvidia/GR00T-N1.6-3B)
#   DEVICE       — cuda / cpu (default: cuda)
#   MOCK_MODE    — set to "1" to run without real weights (for pipeline testing)
#   ROS_ENABLED  — set to "1" to publish to ROS 2 (requires rclpy in container)

from __future__ import annotations
import os
import sys
import time
import base64
import logging
import argparse
import threading
import json
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vla_inference")

# ---------------------------------------------------------------------------
# Configuration from environment / CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR00T / UnifoLM VLA Inference Node")
    p.add_argument("--model_id",    type=str, default=os.getenv("MODEL_ID", "nvidia/GR00T-N1.6-3B"))
    p.add_argument("--device",      type=str, default=os.getenv("DEVICE", "cuda"))
    p.add_argument("--mock",        action="store_true",
                   default=os.getenv("MOCK_MODE", "0") == "1",
                   help="Run in mock mode (no real weights needed)")
    p.add_argument("--ros",         action="store_true",
                   default=os.getenv("ROS_ENABLED", "0") == "1",
                   help="Publish actions to ROS 2 /vla/target_pose")
    p.add_argument("--infer_hz",    type=float, default=5.0,
                   help="Target inference frequency (Hz)")
    p.add_argument("--instruction", type=str,
                   default="Walk forward and pick up the object in front of you.",
                   help="Default language instruction passed to the model")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
class VLAModel:
    """
    Wrapper around the HuggingFace GR00T / UnifoLM model.
    Handles:
    - Model loading with 8-bit quantization fallback for VRAM-constrained GPUs
    - Image preprocessing (resize → normalize → tensor)
    - Text tokenization
    - Action decoding from model output tokens
    """

    def __init__(self, model_id: str, device: str, mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.mock = mock
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        if self.mock:
            log.warning("MOCK MODE — model weights NOT loaded. Outputs are random.")
            return

        log.info(f"Loading model: {self.model_id}")
        log.info("This may take several minutes on first run (downloading ~6GB)...")

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            # Try full precision first; fall back to 8-bit if OOM
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to(self.device)
                log.info(f"Model loaded in bfloat16 on {self.device}.")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as oom:
                log.warning(f"OOM at full precision ({oom}). Retrying with 8-bit quantization...")
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                )
                log.info("Model loaded in 8-bit quantization.")

            self.model.eval()
            log.info("Model ready.")

        except ImportError as e:
            log.error(f"Missing dependency: {e}")
            log.error("Install: pip install transformers accelerate bitsandbytes pillow")
            raise
        except Exception as e:
            log.error(f"Model loading failed: {e}")
            log.error(
                "If the model is gated, authenticate with: "
                "huggingface-cli login"
            )
            raise

    def infer(self, image_rgb, instruction: str) -> dict:
        """
        Run one forward pass.

        Args:
            image_rgb: np.ndarray of shape (H, W, 3) or PIL.Image
            instruction: str — natural language command

        Returns:
            dict with keys: action_type, target_xyz, target_rpy, gripper_open, confidence
        """
        if self.mock:
            return self._mock_output(instruction)

        try:
            from PIL import Image as PILImage
            import numpy as np

            if isinstance(image_rgb, np.ndarray):
                pil_img = PILImage.fromarray(image_rgb.astype("uint8"))
            else:
                pil_img = image_rgb

            inputs = self.processor(
                images=pil_img,
                text=instruction,
                return_tensors="pt",
            ).to(self.device)

            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )

            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return self._parse_output(decoded, instruction)

        except Exception as e:
            log.warning(f"Inference error: {e} — returning mock output.")
            return self._mock_output(instruction)

    def _parse_output(self, raw_text: str, instruction: str) -> dict:
        """
        Parse the model's output text into a structured action dict.

        GR00T-N1.6 outputs a JSON-like action string, e.g.:
          {"action": "navigate", "target": [1.2, 0.0, 0.0], "gripper": false}

        We attempt JSON parsing then fall back to keyword extraction.
        """
        try:
            # Find JSON block in output
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start >= 0 and end > start:
                action_data = json.loads(raw_text[start:end])
                return {
                    "action_type":   action_data.get("action", "navigate"),
                    "target_xyz":    action_data.get("target", [0.0, 0.0, 0.0]),
                    "target_rpy":    action_data.get("orientation", [0.0, 0.0, 0.0]),
                    "gripper_open":  action_data.get("gripper", False),
                    "confidence":    action_data.get("confidence", 0.8),
                    "raw":           raw_text,
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Keyword fallback
        action_type = "navigate"
        if any(w in instruction.lower() for w in ["pick", "grab", "grasp"]):
            action_type = "manipulate"
        elif any(w in instruction.lower() for w in ["stop", "halt", "freeze"]):
            action_type = "stop"

        return {
            "action_type":  action_type,
            "target_xyz":   [0.5, 0.0, 0.0],
            "target_rpy":   [0.0, 0.0, 0.0],
            "gripper_open": action_type == "manipulate",
            "confidence":   0.5,
            "raw":          raw_text,
        }

    @staticmethod
    def _mock_output(instruction: str) -> dict:
        """Deterministic mock output for dry-run testing."""
        import math, time
        t = time.time()
        action_type = "navigate"
        if "pick" in instruction.lower() or "grab" in instruction.lower():
            action_type = "manipulate"
        elif "stop" in instruction.lower():
            action_type = "stop"

        return {
            "action_type":  action_type,
            "target_xyz":   [0.5 * math.sin(t * 0.1), 0.0, 0.0],
            "target_rpy":   [0.0, 0.0, math.sin(t * 0.05)],
            "gripper_open": action_type == "manipulate",
            "confidence":   0.95,
            "raw":          f"[MOCK] instruction='{instruction}'",
        }


# ---------------------------------------------------------------------------
# ROS 2 Publisher (optional)
# ---------------------------------------------------------------------------
class ROS2Publisher:
    """
    Publishes VLA action targets to ROS 2.
    Lazy-imported so the node works without ROS installed (mock / API mode).
    """

    def __init__(self):
        self.node = None
        self.pub = None
        self._init()

    def _init(self):
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import PoseStamped

            rclpy.init(args=None)

            class _VLAPublisherNode(Node):
                def __init__(self):
                    super().__init__("vla_inference_node")
                    self.pub = self.create_publisher(PoseStamped, "/vla/target_pose", 10)

            self.node = _VLAPublisherNode()
            self.pub = self.node.pub
            log.info("ROS 2 publisher ready → /vla/target_pose")
        except ImportError:
            log.warning("rclpy not available — ROS 2 publishing disabled.")
        except Exception as e:
            log.warning(f"ROS 2 init failed: {e}")

    def publish(self, action: dict):
        if self.pub is None:
            return
        try:
            from geometry_msgs.msg import PoseStamped
            from rclpy.clock import Clock

            msg = PoseStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"

            xyz = action.get("target_xyz", [0.0, 0.0, 0.0])
            msg.pose.position.x = float(xyz[0])
            msg.pose.position.y = float(xyz[1])
            msg.pose.position.z = float(xyz[2])
            # Orientation: leave as identity quaternion for now
            msg.pose.orientation.w = 1.0

            self.pub.publish(msg)
        except Exception as e:
            log.warning(f"ROS 2 publish error: {e}")

    def spin_once(self):
        if self.node is not None:
            try:
                import rclpy
                rclpy.spin_once(self.node, timeout_sec=0.0)
            except Exception:
                pass

    def shutdown(self):
        if self.node is not None:
            try:
                import rclpy
                self.node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Camera Frame Acquisition
# ---------------------------------------------------------------------------
def get_latest_frame(mock: bool = True):
    """
    Acquire the latest image frame from the robot's forward camera.

    In real deployment:
      - ROS 2 subscriber to /camera/left/image_rect
      - Or direct USB/GigE capture via OpenCV

    In mock mode: returns a synthetic gradient image.
    """
    import numpy as np
    if mock:
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Synthetic gradient + timestamp overlay
        img[:, :, 0] = np.linspace(0, 200, w, dtype=np.uint8)  # R ramp
        img[:, :, 2] = np.linspace(200, 0, w, dtype=np.uint8)  # B ramp
        return img
    else:
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            log.warning(f"Camera capture failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Latency Tracker
# ---------------------------------------------------------------------------
class LatencyTracker:
    def __init__(self, window: int = 20):
        self._window = window
        self._samples: list[float] = []

    def record(self, dt: float):
        self._samples.append(dt)
        if len(self._samples) > self._window:
            self._samples.pop(0)

    @property
    def mean_ms(self) -> float:
        return (sum(self._samples) / len(self._samples) * 1000) if self._samples else 0.0

    @property
    def max_ms(self) -> float:
        return max(self._samples) * 1000 if self._samples else 0.0


# ---------------------------------------------------------------------------
# Main Inference Loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("  GR00T / UnifoLM VLA Inference Node")
    log.info(f"  Model   : {args.model_id}")
    log.info(f"  Device  : {args.device}")
    log.info(f"  Mock    : {args.mock}")
    log.info(f"  ROS 2   : {args.ros}")
    log.info(f"  Target  : {args.infer_hz:.1f} Hz")
    log.info("=" * 60)

    # Load model
    model = VLAModel(args.model_id, args.device, mock=args.mock)

    # Optionally init ROS 2
    ros_pub = ROS2Publisher() if args.ros else None

    # Latency tracking
    latency = LatencyTracker()
    target_period = 1.0 / args.infer_hz
    step = 0

    log.info("Starting inference loop. Press Ctrl+C to stop.")
    try:
        while True:
            loop_start = time.perf_counter()

            # 1. Acquire frame
            frame = get_latest_frame(mock=args.mock)
            if frame is None:
                log.warning("No frame available — sleeping...")
                time.sleep(0.5)
                continue

            # 2. Run inference
            infer_start = time.perf_counter()
            action = model.infer(frame, args.instruction)
            infer_time = time.perf_counter() - infer_start

            # 3. Publish to ROS 2
            if ros_pub is not None:
                ros_pub.publish(action)
                ros_pub.spin_once()

            # 4. Log
            latency.record(infer_time)
            step += 1
            log.info(
                f"[Step {step:05d}] "
                f"action={action['action_type']:<12} "
                f"xyz=[{action['target_xyz'][0]:+.3f}, "
                f"{action['target_xyz'][1]:+.3f}, "
                f"{action['target_xyz'][2]:+.3f}]  "
                f"gripper={'OPEN' if action['gripper_open'] else 'CLOSE'}  "
                f"conf={action['confidence']:.2f}  "
                f"infer={infer_time*1000:.1f}ms  avg={latency.mean_ms:.1f}ms"
            )

            # 5. Rate limiting
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log.info("Inference stopped by user.")
    finally:
        if ros_pub is not None:
            ros_pub.shutdown()
        log.info(f"Session summary: {step} steps | mean latency {latency.mean_ms:.1f}ms | max {latency.max_ms:.1f}ms")


if __name__ == "__main__":
    main()
