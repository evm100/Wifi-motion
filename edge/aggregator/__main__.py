"""
__main__.py — Entry point for the CSI edge pipeline.

Usage:
    python -m edge.aggregator --config edge/config/pipeline.yaml

Loads configuration, starts the async UDP receiver, wires it through
packet parsing → alignment → DSP pipeline → GPU forwarding.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections import deque
from pathlib import Path

import numpy as np
import yaml

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edge.aggregator.udp_receiver import start_udp_receiver
from edge.aggregator.packet_parser import parse_packet, CSIPacket
from edge.aggregator.aligner import Aligner
from edge.dsp.phase_sanitizer import sanitize_phase_linear
from edge.dsp.amplitude_filter import HampelFilter, ButterFilter
from edge.dsp.baseline import AdaptiveBaseline
from edge.dsp.pca import StreamingCSIPCA
from edge.dsp.feature_extractor import CSIFeatureExtractor
from edge.forwarding.gpu_forwarder import GPUForwarder

logger = logging.getLogger("csi_pipeline")


def load_config(pipeline_path: str, network_path: str | None = None) -> dict:
    """Load and merge pipeline + network YAML configs."""
    config: dict = {}
    with open(pipeline_path) as f:
        config["pipeline"] = yaml.safe_load(f)

    if network_path is None:
        network_path = str(Path(pipeline_path).parent / "network.yaml")
    if Path(network_path).exists():
        with open(network_path) as f:
            config["network"] = yaml.safe_load(f)
    else:
        config["network"] = {}

    return config


class DSPPipeline:
    """Complete real-time DSP pipeline wiring all processing stages."""

    def __init__(self, config: dict) -> None:
        pc = config.get("pipeline", {})
        nc = config.get("network", {})

        self.n_nodes: int = nc.get("n_nodes", 3)
        fs: float = pc.get("csi_sampling_rate_hz", 100.0)

        # DSP components
        hampel_cfg = pc.get("hampel", {})
        self.hampel = HampelFilter(
            window_size=hampel_cfg.get("window_size", 5),
            n_sigma=hampel_cfg.get("n_sigma", 3.0),
        )
        self.hampel_enabled: bool = hampel_cfg.get("enabled", True)

        butter_cfg = pc.get("butterworth", {})
        self.butter = ButterFilter(
            cutoff_hz=butter_cfg.get("cutoff_hz", 10.0),
            fs=fs,
            order=butter_cfg.get("order", 4),
        )

        baseline_cfg = pc.get("baseline", {})
        self.baselines: dict[int, AdaptiveBaseline] = {}
        self._baseline_fast_alpha = baseline_cfg.get("fast_alpha", 0.1)
        self._baseline_slow_alpha = baseline_cfg.get("slow_alpha", 0.001)
        self._baseline_n_required = baseline_cfg.get("calibration_samples", 300)

        pca_cfg = pc.get("pca", {})
        n_sc = pc.get("n_valid_subcarriers", 108)
        self.pca = StreamingCSIPCA(
            n_nodes=self.n_nodes,
            n_subcarriers=n_sc,
            n_components=pca_cfg.get("n_components", 20),
            calibration_size=pca_cfg.get("calibration_size", 500),
        )

        feat_cfg = pc.get("feature_extraction", {})
        self.extractor = CSIFeatureExtractor(
            fs=fs,
            window_size=feat_cfg.get("window_size", 256),
            hop_size=feat_cfg.get("hop_size", 50),
        )

        gpu_addr = nc.get("gpu_address", "tcp://*:5556")
        gpu_hwm = nc.get("gpu_send_hwm", 100)
        self.forwarder = GPUForwarder(gpu_address=gpu_addr, send_hwm=gpu_hwm)

        # Per-node amplitude history for Hampel (needs sliding window)
        self.amp_history: dict[int, deque] = {}

        # PCA score buffer for GPU streaming
        self.pc_buffer: deque = deque(maxlen=500)

        # State
        self.calibration_phase = True
        self.frame_count = 0
        self.hop_size = feat_cfg.get("hop_size", 50)

    def _get_baseline(self, node_id: int) -> AdaptiveBaseline:
        if node_id not in self.baselines:
            self.baselines[node_id] = AdaptiveBaseline(
                fast_alpha=self._baseline_fast_alpha,
                slow_alpha=self._baseline_slow_alpha,
            )
        return self.baselines[node_id]

    def _get_amp_history(self, node_id: int) -> deque:
        if node_id not in self.amp_history:
            self.amp_history[node_id] = deque(maxlen=300)
        return self.amp_history[node_id]

    def process_aligned_group(self, group: dict[int, CSIPacket]) -> None:
        """Process one aligned frame group through the full DSP chain."""
        self.frame_count += 1
        node_amplitudes: dict[int, np.ndarray] = {}

        for node_id, pkt in group.items():
            amp = pkt.amplitude.copy()

            # Phase sanitization (per packet)
            sanitize_phase_linear(pkt.csi_complex)

            # Hampel: accumulate history and filter latest sample
            if self.hampel_enabled:
                history = self._get_amp_history(node_id)
                history.append(amp)
                if len(history) >= 2 * self.hampel.window_size + 1:
                    recent = np.array(list(history))[-(2 * self.hampel.window_size + 1) :]
                    filtered, _ = self.hampel.filter(recent)
                    amp = filtered[-1]

            # Calibration or operation
            baseline = self._get_baseline(node_id)
            if self.calibration_phase:
                done = baseline.calibrate(amp, n_required=self._baseline_n_required)
                node_amplitudes[node_id] = amp

                if done and all(
                    b.calibrated
                    for b in self.baselines.values()
                    if True  # check all created
                ) and len(self.baselines) == self.n_nodes:
                    cal_done = self.pca.add_calibration_frame(node_amplitudes)
                    if cal_done:
                        self.calibration_phase = False
                        logger.info(
                            "Calibration complete. Top-5 variance explained: %s",
                            self.pca.explained_variance_ratio[:5]
                            if self.pca.explained_variance_ratio is not None
                            else "N/A",
                        )
            else:
                dynamic = baseline.remove_static(amp)
                node_amplitudes[node_id] = dynamic

        # PCA + GPU streaming (post-calibration only)
        if not self.calibration_phase:
            pc_scores = self.pca.transform(node_amplitudes)
            if pc_scores is not None:
                self.pc_buffer.append(pc_scores)

                # Stream to GPU every hop_size frames
                if self.frame_count % self.hop_size == 0 and len(self.pc_buffer) >= 50:
                    recent_pcs = np.array(list(self.pc_buffer)[-50:])
                    motion_energy = float(np.sum(np.var(recent_pcs, axis=0)))

                    window = np.array(list(self.pc_buffer))
                    self.forwarder.send_tensor(
                        window,
                        metadata={
                            "frame": self.frame_count,
                            "motion_energy": motion_energy,
                            "n_pcs": len(pc_scores),
                        },
                    )

    def close(self) -> None:
        self.forwarder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CSI Edge Pipeline")
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "edge" / "config" / "pipeline.yaml"),
        help="Path to pipeline.yaml config file",
    )
    parser.add_argument(
        "--network-config",
        default=None,
        help="Path to network.yaml (default: same dir as pipeline.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config, args.network_config)
    nc = config.get("network", {})

    pipeline = DSPPipeline(config)

    aligner = Aligner(
        n_nodes=pipeline.n_nodes,
        stale_threshold=nc.get("aligner_stale_threshold", 50),
        on_aligned=pipeline.process_aligned_group,
    )

    def on_raw_packet(data: bytes, addr: tuple[str, int]) -> None:
        try:
            packet = parse_packet(data)
            aligner.add_packet(packet)
        except ValueError as exc:
            logger.debug("Bad packet from %s: %s", addr, exc)

    async def run() -> None:
        udp_host = nc.get("udp_host", "0.0.0.0")
        udp_port = nc.get("udp_port", 5005)
        transport, protocol = await start_udp_receiver(
            on_packet=on_raw_packet,
            host=udp_host,
            port=udp_port,
        )

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        logger.info("Pipeline running. Press Ctrl+C to stop.")
        await stop.wait()

        transport.close()
        pipeline.close()
        logger.info(
            "Shutdown. %d groups processed, %d packets received.",
            aligner.stats.groups_emitted,
            aligner.stats.packets_received,
        )

    asyncio.run(run())


if __name__ == "__main__":
    main()
