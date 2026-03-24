"""ZMQ inference server — receives spectrograms from Pi, classifies, logs results."""

from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.classifier import RealtimeCSIClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("csi-inference")


def main() -> None:
    model_path = os.environ.get("MODEL_PATH", "checkpoints/best_resnet.pt")
    zmq_address = os.environ.get("ZMQ_ADDRESS", "tcp://192.168.4.1:5556")
    n_channels = int(os.environ.get("N_CHANNELS", "30"))
    n_classes = int(os.environ.get("N_CLASSES", "7"))
    device = os.environ.get("DEVICE", "cuda")

    log.info("Loading model from %s", model_path)
    classifier = RealtimeCSIClassifier(
        model_path=model_path if Path(model_path).exists() else None,
        n_input_channels=n_channels,
        n_classes=n_classes,
        device=device,
    )
    log.info("Model loaded on %s. Connecting to %s", classifier.device, zmq_address)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(zmq_address)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 5000)

    log.info("Inference server running. Waiting for data...")

    try:
        while True:
            try:
                raw = sub.recv()
            except zmq.Again:
                continue

            try:
                msg = pickle.loads(raw)
            except Exception:
                log.warning("Failed to unpickle message, skipping")
                continue

            tensor = msg.get("tensor")
            metadata = msg.get("metadata", {})

            if tensor is None:
                log.warning("Message missing 'tensor' key, skipping")
                continue

            activity, confidence, latency_ms = classifier.predict(tensor)

            log.info(
                "seq=%s | prediction=%-10s confidence=%.3f latency=%.1fms",
                metadata.get("seq_num", "?"),
                activity,
                confidence,
                latency_ms,
            )

    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        sub.close()
        ctx.term()


if __name__ == "__main__":
    main()
