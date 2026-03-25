"""
collector.py -- Data collection session manager.

Reuses the recording logic from tools/collect_data.py but driven by the
web UI instead of terminal prompts.  The monitor's UDP socket is shared:
when a collection is active, incoming packets are also written to disk.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from proto.csi_packet import CSIPacketHeader, HEADER_SIZE, CSI_MAGIC
from proto.constants import CSI_PACKET_SIZE


class SessionState(str, Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"


DEFAULT_ACTIVITIES = [
    "empty", "walking", "sitting", "standing",
    "falling", "gesture", "breathing",
]


@dataclass
class ActiveCapture:
    activity: str
    repetition: int
    start_time: float  # monotonic
    duration: float
    bin_path: Path
    file: object = None  # open file handle
    packet_count: int = 0
    node_ids_seen: set = field(default_factory=set)


class Collector:
    """Manages data collection sessions triggered from the web UI."""

    def __init__(self, output_dir: str = "data/sessions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state: SessionState = SessionState.IDLE
        self.session_id: str | None = None
        self._capture: ActiveCapture | None = None
        self._countdown_end: float = 0.0
        self._all_metadata: list[dict] = []

        # Queue of activities to record: list of (activity, rep)
        self._queue: list[tuple[str, int]] = []

    def get_status(self) -> dict:
        status: dict = {"state": self.state.value, "session_id": self.session_id}
        if self.state == SessionState.COUNTDOWN:
            remaining = max(0, self._countdown_end - time.monotonic())
            status["countdown_remaining"] = round(remaining, 1)
            if self._queue:
                status["next_activity"] = self._queue[0][0]
                status["next_repetition"] = self._queue[0][1]
        elif self.state == SessionState.RECORDING and self._capture:
            elapsed = time.monotonic() - self._capture.start_time
            status["activity"] = self._capture.activity
            status["repetition"] = self._capture.repetition
            status["elapsed"] = round(elapsed, 1)
            status["duration"] = self._capture.duration
            status["packet_count"] = self._capture.packet_count
            status["node_ids"] = sorted(self._capture.node_ids_seen)
        status["queue_remaining"] = len(self._queue)
        status["captures_done"] = len(self._all_metadata)
        return status

    def start_session(
        self,
        activities: list[str] | None = None,
        duration: float = 30.0,
        repetitions: int = 1,
        countdown: float = 5.0,
    ) -> dict:
        if self.state != SessionState.IDLE:
            return {"error": "Session already active"}

        if activities is None:
            activities = DEFAULT_ACTIVITIES

        self.session_id = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")
        self._all_metadata = []
        self._queue = [
            (act, rep)
            for act in activities
            for rep in range(1, repetitions + 1)
        ]
        self._duration = duration
        self._countdown_secs = countdown

        # Start countdown for first activity.
        self._begin_countdown()
        return {"session_id": self.session_id, "total_captures": len(self._queue)}

    def _begin_countdown(self) -> None:
        self.state = SessionState.COUNTDOWN
        self._countdown_end = time.monotonic() + self._countdown_secs

    def start_next_capture(self) -> None:
        """Called by the app's tick loop when countdown finishes."""
        if not self._queue:
            self._finish_session()
            return

        activity, rep = self._queue.pop(0)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = f"{self.session_id}_{activity}_rep{rep:02d}_{ts}"
        bin_path = self.output_dir / f"{base}.bin"

        cap = ActiveCapture(
            activity=activity,
            repetition=rep,
            start_time=time.monotonic(),
            duration=self._duration,
            bin_path=bin_path,
        )
        cap.file = open(bin_path, "wb")
        self._capture = cap
        self.state = SessionState.RECORDING

    def handle_packet(self, data: bytes) -> None:
        """Feed raw UDP data when a recording is active."""
        if self.state != SessionState.RECORDING or self._capture is None:
            return
        if self._capture.file is None:
            return

        if len(data) < HEADER_SIZE:
            return
        try:
            hdr = CSIPacketHeader.from_bytes(data)
        except ValueError:
            return

        self._capture.node_ids_seen.add(hdr.node_id)
        length = len(data)
        self._capture.file.write(length.to_bytes(2, "little"))
        self._capture.file.write(data)
        self._capture.packet_count += 1

    def tick(self) -> None:
        """Called periodically (~1 Hz) to advance session state machine."""
        if self.state == SessionState.COUNTDOWN:
            if time.monotonic() >= self._countdown_end:
                self.start_next_capture()

        elif self.state == SessionState.RECORDING and self._capture:
            elapsed = time.monotonic() - self._capture.start_time
            if elapsed >= self._capture.duration:
                self._end_capture()
                if self._queue:
                    self._begin_countdown()
                else:
                    self._finish_session()

    def skip_current(self) -> dict:
        """Skip the currently recording activity."""
        if self.state == SessionState.RECORDING and self._capture:
            self._end_capture()
            if self._queue:
                self._begin_countdown()
            else:
                self._finish_session()
            return {"skipped": True}
        return {"skipped": False}

    def stop_session(self) -> dict:
        """Stop the entire session."""
        if self._capture and self._capture.file:
            self._end_capture()
        self._finish_session()
        return {"stopped": True}

    def _end_capture(self) -> None:
        cap = self._capture
        if cap is None:
            return
        if cap.file:
            cap.file.close()
            cap.file = None

        actual_duration = time.monotonic() - cap.start_time
        meta = {
            "session_id": self.session_id,
            "activity": cap.activity,
            "repetition": cap.repetition,
            "duration_seconds": round(actual_duration, 3),
            "packet_count": cap.packet_count,
            "node_ids": sorted(cap.node_ids_seen),
            "node_count": len(cap.node_ids_seen),
            "binary_file": cap.bin_path.name,
        }
        # Write JSON sidecar
        meta_path = cap.bin_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._all_metadata.append(meta)
        self._capture = None

    def _finish_session(self) -> None:
        if self.session_id and self._all_metadata:
            summary_path = self.output_dir / f"{self.session_id}_summary.json"
            summary = {
                "session_id": self.session_id,
                "total_captures": len(self._all_metadata),
                "total_packets": sum(m["packet_count"] for m in self._all_metadata),
                "captures": self._all_metadata,
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

        self.state = SessionState.IDLE
        self._queue = []
        self._capture = None

    def list_sessions(self) -> list[dict]:
        """List past session summaries from disk."""
        sessions = []
        for p in sorted(self.output_dir.glob("session_*_summary.json"), reverse=True):
            try:
                with open(p) as f:
                    sessions.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        return sessions
