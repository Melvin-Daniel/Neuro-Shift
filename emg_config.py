"""
Load EMG serial/model/live-gate settings from emg_config.json with safe defaults.

Serial port: only one process may open it at a time. Close Arduino Serial Monitor /
Serial Plotter before running live_emg_demo.py, collect_data.py, or any LAN bridge
that reads the same COM port.

Environment overrides (optional):
  EMG_SERIAL_PORT  — wins over JSON serial.port
  EMG_MODEL_PATH   — wins over JSON model.path (same as live demo historically)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONFIG_FILENAME = "emg_config.json"

DEFAULT_SERIAL: dict[str, Any] = {"port": "COM4", "baud": 9600}
DEFAULT_MODEL: dict[str, Any] = {"path": "emg_rf_model.pkl"}
DEFAULT_SERIAL_RECONNECT: dict[str, Any] = {
    "initial_backoff_s": 0.5,
    "max_backoff_s": 5.0,
    "stale_frame_warn_s": 3.0,
    "stale_warn_interval_s": 5.0,
}
DEFAULT_LIVE: dict[str, Any] = {
    "window_size": 100,
    "step_size": 10,
    "arm_timeout_s": 0.0,
    "stay_armed_after_jaw": True,
    "consensus_n_jaw": 4,
    # Need this many jaw_eligible ticks in the last consensus_n_jaw windows (sliding vote).
    "consensus_jaw_min_votes": 3,
    # While ARMED: also fire after this many jaw_eligible steps (not reset by rest/eyebrow in between).
    "consensus_jaw_armed_accum": 4,
    "min_proba_margin_jaw": 0.20,
    "confidence_gate_jaw": 0.80,
    "consensus_n_eyebrow": 2,
    "confidence_gate_eyebrow": 0.58,
    "min_proba_margin_eyebrow": 0.06,
    "eyebrow_double_tap_window_s": 1.6,
    "eyebrow_tap_debounce_s": 0.7,
    "eyebrow_toggle_cooldown_s": 1.8,
    "cooldown_s": 2.0,
    "use_activity_gate": True,
    "use_activity_gate_eyebrow": True,
    "use_activity_gate_jaw": True,
    "rest_baseline_conf": 0.45,
    "activity_min_baseline_samples": 8,
    "activity_rms_multiplier_jaw": 1.52,
    "activity_rms_multiplier_eyebrow": 1.12,
    # If brow RMS gate fails but the model is this sure it's eyebrow_raise, still count toward ARM (reduces stuck act=N).
    "brow_activity_bypass_min_conf": 0.9,
    # Same for jaw: model often high-conf jaw_clench while RMS vs rest baseline stays low (dual-channel / noise).
    "jaw_activity_bypass_min_conf": 0.9,
    "activity_baseline_max": 80,
    "use_brow_tap_detector": True,
    "brow_tap_rms_multiplier": 1.35,
    "brow_tap_min_p2p": 45.0,
    # Jaw command while ARMED: model often predicts rest during clench (dual-channel quirk).
    "use_jaw_tap_detector": True,
    "jaw_tap_only_when_armed": True,
    "jaw_tap_rms_multiplier": 1.48,
    "jaw_tap_min_p2p": 52.0,
}


def load_emg_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (ROOT / CONFIG_FILENAME)
    serial = DEFAULT_SERIAL.copy()
    model = DEFAULT_MODEL.copy()
    reconnect = DEFAULT_SERIAL_RECONNECT.copy()
    live = DEFAULT_LIVE.copy()

    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("serial"), dict):
            serial.update(data["serial"])
        if isinstance(data.get("model"), dict):
            model.update(data["model"])
        if isinstance(data.get("serial_reconnect"), dict):
            reconnect.update(data["serial_reconnect"])
        if isinstance(data.get("live"), dict):
            live.update(data["live"])

    env_port = (os.environ.get("EMG_SERIAL_PORT") or "").strip()
    if env_port:
        serial["port"] = env_port
    env_model = (os.environ.get("EMG_MODEL_PATH") or "").strip()
    if env_model:
        model["path"] = env_model

    baud = serial.get("baud", 9600)
    try:
        serial["baud"] = int(baud)
    except (TypeError, ValueError):
        serial["baud"] = 9600

    for key in ("initial_backoff_s", "max_backoff_s", "stale_frame_warn_s", "stale_warn_interval_s"):
        if key in reconnect:
            try:
                reconnect[key] = float(reconnect[key])
            except (TypeError, ValueError):
                reconnect[key] = DEFAULT_SERIAL_RECONNECT[key]

    for key, default in DEFAULT_LIVE.items():
        if key not in live:
            live[key] = default
    for k in (
        "window_size",
        "step_size",
        "consensus_n_jaw",
        "consensus_jaw_min_votes",
        "consensus_jaw_armed_accum",
        "consensus_n_eyebrow",
        "activity_min_baseline_samples",
        "activity_baseline_max",
    ):
        try:
            live[k] = int(live[k])
        except (TypeError, ValueError):
            live[k] = DEFAULT_LIVE[k]
    for k in (
        "arm_timeout_s",
        "confidence_gate_jaw",
        "confidence_gate_eyebrow",
        "min_proba_margin_jaw",
        "min_proba_margin_eyebrow",
        "eyebrow_double_tap_window_s",
        "eyebrow_tap_debounce_s",
        "eyebrow_toggle_cooldown_s",
        "cooldown_s",
        "rest_baseline_conf",
        "activity_rms_multiplier_jaw",
        "activity_rms_multiplier_eyebrow",
        "brow_activity_bypass_min_conf",
        "jaw_activity_bypass_min_conf",
        "brow_tap_rms_multiplier",
        "brow_tap_min_p2p",
        "jaw_tap_rms_multiplier",
        "jaw_tap_min_p2p",
    ):
        try:
            live[k] = float(live[k])
        except (TypeError, ValueError):
            live[k] = DEFAULT_LIVE[k]
    for k in (
        "stay_armed_after_jaw",
        "use_activity_gate",
        "use_activity_gate_eyebrow",
        "use_activity_gate_jaw",
        "use_brow_tap_detector",
        "use_jaw_tap_detector",
        "jaw_tap_only_when_armed",
    ):
        live[k] = bool(live[k])

    nj = int(live["consensus_n_jaw"])
    mv = int(live.get("consensus_jaw_min_votes", nj))
    mv = max(1, min(mv, nj))
    live["consensus_jaw_min_votes"] = mv
    aa = int(live.get("consensus_jaw_armed_accum", 2))
    live["consensus_jaw_armed_accum"] = max(0, aa)

    return {"serial": serial, "model": model, "serial_reconnect": reconnect, "live": live}
