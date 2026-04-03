import time
import pickle
from collections import deque
import os
from pathlib import Path

import numpy as np
import serial
import urllib.request
from serial.serialutil import SerialException

from emg_config import load_emg_config

# Live gate defaults live in emg_config.json (see emg_config.DEFAULT_LIVE).


def extract_features(signal: np.ndarray) -> list[float]:
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    peak = float(np.max(signal) - np.min(signal))
    abs_mean = float(np.mean(np.abs(signal)))
    energy = float(np.sum(signal**2))
    zero_cross = float(np.sum(np.diff(np.sign(signal)) != 0))
    return [mean, std, peak, abs_mean, energy, zero_cross]


def extract_features_dual(jaw: np.ndarray, brow: np.ndarray) -> list[float]:
    """12 features: 6 from jaw channel + 6 from brow channel (two AD8232)."""
    return extract_features(jaw) + extract_features(brow)


def window_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal**2)))


def parse_line(line: str) -> tuple[int, int | None] | None:
    """Returns (jaw, brow) or (single, None) for legacy 2-field lines."""
    line = line.strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    try:
        if len(parts) >= 3:
            return int(parts[1]), int(parts[2])
        if len(parts) >= 2:
            return int(parts[1]), None
    except Exception:
        return None
    return None


def best_second_margin(proba: np.ndarray) -> tuple[int, float, float]:
    order = np.argsort(proba)[::-1]
    best_i = int(order[0])
    p_best = float(proba[best_i])
    p_second = float(proba[order[1]]) if len(order) > 1 else 0.0
    return best_i, p_best, p_best - p_second


def confidence_floor_for_label(
    label: str,
    on_label: str,
    off_label: str,
    gate_jaw: float,
    gate_brow: float,
) -> float:
    if label == on_label:
        return gate_jaw
    if label == off_label:
        return gate_brow
    return 0.0


def plug_is_on_from_status(status: object) -> bool | None:
    """Parse tinytuya OutletDevice.status() dict; return None if unknown."""
    if not isinstance(status, dict):
        return None
    if status.get("Error") is not None:
        return None
    dps = status.get("dps")
    if not isinstance(dps, dict):
        return None
    for key in ("1", "20", 1, 20):
        if key in dps:
            v = dps[key]
            if v is True or v == 1 or v == "1" or v == "true":
                return True
            if v is False or v == 0 or v == "0" or v == "false":
                return False
    return None


def _tuya_control_ok(result: object) -> bool:
    """tinytuya returns a dict; Err/Error means the device rejected the command."""
    if result is None:
        return False
    if not isinstance(result, dict):
        return True
    if result.get("Error") is not None:
        return False
    if result.get("Err") is not None:
        return False
    return True


def main() -> int:
    try:
        from dotenv import load_dotenv

        # Existing OS env wins over .env (so PowerShell $env:TUYA_* tests apply without editing .env).
        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    except ImportError:
        pass

    cfg = load_emg_config()
    port = cfg["serial"]["port"]
    baud = cfg["serial"]["baud"]
    sr = cfg["serial_reconnect"]
    L = cfg["live"]

    window_size = L["window_size"]
    step_size = L["step_size"]

    on_label = "jaw_clench"
    off_label = "eyebrow_raise"
    idle_label = "rest"

    model_path = cfg["model"]["path"]

    ifttt_key = os.environ.get("IFTTT_KEY", "").strip()
    ifttt_event_on = os.environ.get("IFTTT_EVENT_ON", "neuro_shift_on").strip()
    ifttt_event_off = os.environ.get("IFTTT_EVENT_OFF", "neuro_shift_off").strip()

    def trigger_ifttt(event_name: str) -> bool:
        if not ifttt_key:
            return False
        url = f"https://maker.ifttt.com/trigger/{event_name}/with/key/{ifttt_key}"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    tuya_id = os.environ.get("TUYA_DEVICE_ID", "").strip()
    tuya_key = os.environ.get("TUYA_LOCAL_KEY", "").strip()
    tuya_ip = os.environ.get("TUYA_IP", "").strip()
    tuya_ver = os.environ.get("TUYA_VERSION", "3.3").strip() or "3.3"
    tuya_ready = bool(tuya_id and tuya_key and tuya_ip)
    tuya_switch_dps = int(os.environ.get("TUYA_SWITCH_DPS", "1") or "1")
    tuya_kind = os.environ.get("TUYA_DEVICE_TYPE", "outlet").strip().lower()
    tuya_verbose = os.environ.get("TUYA_VERBOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    tuya_persist = os.environ.get("TUYA_PERSIST", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    tuya_device = None
    tuya_bulb_detected = False
    tuya_working_ver: float | None = None  # set after a successful command (protocol auto-match)
    last_known_plug_on: bool | None = None

    def _tuya_create(protocol_ver: float | None = None):
        nonlocal tuya_device, tuya_bulb_detected
        import tinytuya

        if protocol_ver is not None:
            pv = float(protocol_ver)
        elif tuya_working_ver is not None:
            pv = tuya_working_ver
        else:
            pv = float(tuya_ver)
        if tuya_kind == "bulb":
            tuya_device = tinytuya.BulbDevice(tuya_id, tuya_ip, tuya_key)
        else:
            tuya_device = tinytuya.OutletDevice(tuya_id, tuya_ip, tuya_key)
        tuya_device.set_version(pv)
        if tuya_persist:
            tuya_device.set_socketPersistent(True)
        sw = float(os.environ.get("TUYA_SEND_WAIT", "0") or "0")
        if sw > 0:
            tuya_device.set_sendWait(sw)
        if tuya_kind == "bulb" and not tuya_bulb_detected:
            try:
                tuya_device.detect_bulb()
                tuya_bulb_detected = True
            except Exception:
                pass

    def _tuya_protocol_candidates() -> list[float]:
        """Env version first, then common alternates (914 is often wrong version)."""
        out: list[float] = []

        def add(v: float) -> None:
            if not any(abs(v - o) < 1e-9 for o in out):
                out.append(v)

        if tuya_working_ver is not None:
            add(tuya_working_ver)
        add(float(tuya_ver))
        for v in (3.3, 3.4, 3.5):
            add(float(v))
        return out

    def tuya_turn(on: bool) -> bool:
        nonlocal tuya_device, tuya_working_ver
        if not tuya_ready:
            return False
        try:
            import tinytuya
        except ImportError:
            print("  Tuya: install tinytuya: pip install tinytuya")
            return False
        try:
            last_res: object = None
            for ver_try in _tuya_protocol_candidates():
                tuya_device = None
                _tuya_create(protocol_ver=ver_try)
                assert tuya_device is not None
                if tuya_kind == "bulb":
                    res = tuya_device.turn_on() if on else tuya_device.turn_off()
                else:
                    res = (
                        tuya_device.turn_on(switch=tuya_switch_dps)
                        if on
                        else tuya_device.turn_off(switch=tuya_switch_dps)
                    )
                last_res = res
                if _tuya_control_ok(res):
                    tuya_working_ver = ver_try
                    break
                err = res.get("Err") if isinstance(res, dict) else None
                if str(err) != "914":
                    hint = ""
                    print(f"  Tuya: FAILED (device rejected command: {res!r}){hint}")
                    return False
            else:
                res = last_res
                err = res.get("Err") if isinstance(res, dict) else None
                hint = ""
                if str(err) == "914":
                    kl = len(tuya_key)
                    hint = (
                        " (Tuya local key must be exactly 16 characters; "
                        f"yours has {kl}. If length is wrong, re-copy the Local Key from "
                        "Tuya IoT Platform → Cloud → your project → Devices. "
                        "Also confirm Device ID, LAN IP, and that the key was not rotated.)"
                    )
                print(f"  Tuya: FAILED (device rejected command: {res!r}){hint}")
                return False
            time.sleep(0.2)
            st = tuya_device.status()
            parsed = plug_is_on_from_status(st)
            if tuya_verbose and isinstance(st, dict):
                print(f"  Tuya: status dps={st.get('dps')!r}")
            if parsed is not None and parsed != on:
                print(
                    "  Tuya: WARN sent "
                    f"{'ON' if on else 'OFF'} but device reports "
                    f"{'ON' if parsed else 'OFF'} — try TUYA_SWITCH_DPS=20 "
                    "or TUYA_DEVICE_TYPE=bulb (see .env.example)"
                )
            return True
        except Exception as e:
            print(f"  Tuya: FAILED ({e})")
            return False

    def tuya_query_on() -> bool | None:
        nonlocal tuya_device
        if not tuya_ready:
            return None
        try:
            import tinytuya
        except ImportError:
            return None
        try:
            if tuya_device is None:
                _tuya_create()
            assert tuya_device is not None
            st = tuya_device.status()
            return plug_is_on_from_status(st)
        except Exception:
            return None

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    n_feat = int(
        getattr(clf, "n_features_in_", None)
        or getattr(clf, "n_features_", 6)
    )
    use_dual = n_feat >= 12
    if use_dual:
        print(f"- Model expects {n_feat} features (dual-channel jaw+brow).")
    else:
        print(
            f"- Model expects {n_feat} features (single-channel). "
            "Retrain with dual CSVs after wiring A0+A1, or use old sketch (one value)."
        )

    session_start = time.time()

    def tmsg(msg: str) -> str:
        return f"[t={time.time() - session_start:.1f}s] {msg}"

    ser: serial.Serial | None = None
    backoff = float(sr["initial_backoff_s"])
    last_parse_ok = time.time()
    last_stale_warn = 0.0

    buf_jaw: deque[int] = deque(maxlen=window_size)
    buf_brow: deque[int] = deque(maxlen=window_size)
    since_last_pred = 0
    last_action_at = 0.0
    last_eyebrow_tap_at = 0.0
    last_eyebrow_toggle_at = 0.0
    eyebrow_taps = 0
    eyebrow_tap_expires_at = 0.0

    consensus_jaw: deque[str | None] = deque(maxlen=L["consensus_n_jaw"])
    consensus_brow: deque[str | None] = deque(maxlen=L["consensus_n_eyebrow"])
    rest_rms_baseline_jaw: deque[float] = deque(maxlen=L["activity_baseline_max"])
    rest_rms_baseline_brow: deque[float] = deque(maxlen=L["activity_baseline_max"])

    armed = False
    armed_until = 0.0
    armed_jaw_accum = 0

    def clear_consensus() -> None:
        nonlocal armed_jaw_accum
        consensus_jaw.clear()
        consensus_brow.clear()
        armed_jaw_accum = 0

    def reset_after_serial_loss() -> None:
        nonlocal armed, since_last_pred, last_eyebrow_tap_at, last_eyebrow_toggle_at
        nonlocal eyebrow_taps, eyebrow_tap_expires_at, armed_until, armed_jaw_accum
        armed = False
        armed_until = 0.0
        buf_jaw.clear()
        buf_brow.clear()
        since_last_pred = 0
        last_eyebrow_tap_at = 0.0
        last_eyebrow_toggle_at = 0.0
        eyebrow_taps = 0
        eyebrow_tap_expires_at = 0.0
        clear_consensus()
        rest_rms_baseline_jaw.clear()
        rest_rms_baseline_brow.clear()

    print("Neuro-Shift live EMG demo (armed confirmation + jaw command)")
    print(f"- Config: emg_config.json (port={port!r}, model={model_path})")
    print(f"- Port: {port} @ {baud} (see sketch_oct3a/sketch_oct3a.ino for timing)")
    print(f"- Window: {window_size} samples, step: {step_size}")
    print(
        f"- Gates: jaw>={L['confidence_gate_jaw']:.0%} margin>={L['min_proba_margin_jaw']:.2f} "
        f"consensus={L['consensus_n_jaw']}w (need {L['consensus_jaw_min_votes']} jaw) | "
        f"armed jaw hits: {L['consensus_jaw_armed_accum']} | "
        f"eyebrow>={L['confidence_gate_eyebrow']:.0%} margin>={L['min_proba_margin_eyebrow']:.2f} "
        f"consensus={L['consensus_n_eyebrow']}w"
    )
    jaw_act = L["use_activity_gate"] and L["use_activity_gate_jaw"]
    brow_gate = L["use_activity_gate"] and L["use_activity_gate_eyebrow"]
    bypass_b = float(L.get("brow_activity_bypass_min_conf", 0.0))
    bypass_j = float(L.get("jaw_activity_bypass_min_conf", 0.0))
    brow_note = (
        f" (high-conf bypass ≥{bypass_b:.0%} if RMS fails)"
        if brow_gate and bypass_b > 0
        else ""
    )
    jaw_note = (
        f" (high-conf bypass ≥{bypass_j:.0%} if RMS fails)"
        if jaw_act and bypass_j > 0
        else ""
    )
    print(
        f"- Activity gate on jaw: {'ON' if jaw_act else 'OFF'}{jaw_note} | "
        f"on eyebrow: {'ON' if brow_gate else 'OFF'}{brow_note}"
    )
    arm_t_msg = "disabled" if L["arm_timeout_s"] <= 0 else f"{L['arm_timeout_s']:.1f}s"
    print(
        f"- Arm timeout: {arm_t_msg} | Stay armed after jaw: {L['stay_armed_after_jaw']} | "
        f"Cooldown: {L['cooldown_s']:.1f}s"
    )
    print(
        f"- RMS gate params: jaw_mult={L['activity_rms_multiplier_jaw']}, "
        f"brow_mult={L['activity_rms_multiplier_eyebrow']}, "
        f"min baseline samples={L['activity_min_baseline_samples']}"
    )
    if use_dual:
        print(
            f"- Tap detectors: brow={'ON' if L['use_brow_tap_detector'] else 'OFF'} "
            f"(rms×{L['brow_tap_rms_multiplier']}, p2p≥{L['brow_tap_min_p2p']}) | "
            f"jaw={'ON' if L['use_jaw_tap_detector'] else 'OFF'} "
            f"(rms×{L['jaw_tap_rms_multiplier']}, p2p≥{L['jaw_tap_min_p2p']}, "
            f"only_when_armed={L['jaw_tap_only_when_armed']})"
        )
    if tuya_ready:
        tuya_line = f"- Tuya local: enabled (IP {tuya_ip}, version {tuya_ver}, type={tuya_kind}"
        if tuya_kind != "bulb":
            tuya_line += f", switch_dps={tuya_switch_dps}"
        tuya_line += ")"
        print(tuya_line)
        if len(tuya_key) != 16:
            print(
                f"- Tuya: WARNING — local key is {len(tuya_key)} character(s); "
                "Tuya expects 16. A truncated key causes Err 914. Re-copy from Tuya IoT (Device → local_key)."
            )
    else:
        print("- Tuya local: disabled (set TUYA_DEVICE_ID, TUYA_LOCAL_KEY, TUYA_IP)")
    if ifttt_key:
        print(f"- IFTTT: enabled (events: {ifttt_event_on}, {ifttt_event_off})")
    else:
        print("- IFTTT: disabled (set IFTTT_KEY for cloud webhook fallback)")
    print("Flow:")
    print(f"  1. {off_label} (eyebrow) -> ARM.")
    print(f"  2. While [ARMED]: {on_label} (jaw) -> toggles plug. Eyebrow again -> DISARM (cancel).")
    if L["stay_armed_after_jaw"]:
        print("  3. After a successful jaw toggle you stay ARMED — jaw again toggles OFF/ON without a new eyebrow.")
    else:
        print("  3. After jaw you return to IDLE — eyebrow again before the next jaw.")
    print("  4. Jaw while IDLE does nothing.")
    print("\nStart performing gestures. Press Ctrl+C to stop.\n")

    try:
        while True:
            now = time.time()
            if L["arm_timeout_s"] > 0 and armed and now > armed_until:
                armed = False
                print(tmsg("ARM timeout -> IDLE (disarmed)"))

            if ser is None or not ser.is_open:
                try:
                    print(tmsg(f"SERIAL: connecting {port} @ {baud}..."))
                    ser = serial.Serial(port, baud, timeout=1)
                    time.sleep(2.0)
                    backoff = float(sr["initial_backoff_s"])
                    last_parse_ok = time.time()
                except (PermissionError, SerialException, OSError) as e:
                    print(tmsg(f"SERIAL: open failed ({e!r}); retry in {backoff:.1f}s"))
                    time.sleep(backoff)
                    backoff = min(backoff * 2, float(sr["max_backoff_s"]))
                    ser = None
                    continue

            try:
                raw_b = ser.readline()
            except SerialException as e:
                print(tmsg(f"SERIAL: read error {e!r}; reconnecting"))
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                reset_after_serial_loss()
                time.sleep(float(sr["initial_backoff_s"]))
                continue

            if not raw_b:
                if (
                    now - last_parse_ok > float(sr["stale_frame_warn_s"])
                    and now - last_stale_warn > float(sr["stale_warn_interval_s"])
                ):
                    print(
                        tmsg(
                            "SERIAL: no valid frames (check USB / close Arduino Serial Monitor)"
                        )
                    )
                    last_stale_warn = now
                continue

            raw = raw_b.decode("utf-8", errors="replace")
            parsed = parse_line(raw)
            if parsed is None:
                continue
            last_parse_ok = time.time()
            jaw_v, brow_v = parsed

            if use_dual:
                if brow_v is None:
                    continue
                buf_jaw.append(jaw_v)
                buf_brow.append(brow_v)
            else:
                buf_jaw.append(jaw_v)

            since_last_pred += 1

            if len(buf_jaw) < window_size:
                continue
            if use_dual and len(buf_brow) < window_size:
                continue
            if since_last_pred < step_size:
                continue
            since_last_pred = 0

            x_j = np.array(list(buf_jaw), dtype=float)
            if use_dual:
                x_b = np.array(list(buf_brow), dtype=float)
                feats = np.array(
                    extract_features_dual(x_j, x_b), dtype=float
                ).reshape(1, -1)
                jaw_rms = window_rms(x_j)
                brow_rms = window_rms(x_b)
            else:
                feats = np.array(extract_features(x_j), dtype=float).reshape(1, -1)
                jaw_rms = window_rms(x_j)
                brow_rms = jaw_rms

            proba = clf.predict_proba(feats)[0]
            classes = list(clf.classes_)
            best_i, conf, margin = best_second_margin(proba)
            label = str(classes[best_i])

            if label == idle_label and float(proba[best_i]) >= L["rest_baseline_conf"]:
                rest_rms_baseline_jaw.append(jaw_rms)
                rest_rms_baseline_brow.append(brow_rms)

            floor = confidence_floor_for_label(
                label,
                on_label,
                off_label,
                L["confidence_gate_jaw"],
                L["confidence_gate_eyebrow"],
            )
            jaw_margin_ok = (
                margin >= L["min_proba_margin_jaw"] if L["min_proba_margin_jaw"] > 0 else True
            )
            brow_margin_ok = (
                margin >= L["min_proba_margin_eyebrow"]
                if L["min_proba_margin_eyebrow"] > 0
                else True
            )
            conf_ok = conf >= floor if label in (on_label, off_label) else False

            activity_ok_jaw = True
            if (
                L["use_activity_gate"]
                and L["use_activity_gate_jaw"]
                and len(rest_rms_baseline_jaw) >= L["activity_min_baseline_samples"]
            ):
                med = float(np.median(np.array(rest_rms_baseline_jaw, dtype=float)))
                if med > 0:
                    activity_ok_jaw = jaw_rms >= med * L["activity_rms_multiplier_jaw"]

            jaw_bypass = float(L.get("jaw_activity_bypass_min_conf", 0.0))
            jaw_activity_effective = activity_ok_jaw
            if (
                L["use_activity_gate"]
                and L["use_activity_gate_jaw"]
                and not activity_ok_jaw
                and label == on_label
                and jaw_bypass > 0
                and conf >= jaw_bypass
            ):
                # Clench often fails RMS vs rest baseline (noise, electrode, dual-channel); trust strong jaw pred.
                floor_j = confidence_floor_for_label(
                    on_label,
                    on_label,
                    off_label,
                    L["confidence_gate_jaw"],
                    L["confidence_gate_eyebrow"],
                )
                if conf >= floor_j and jaw_margin_ok:
                    jaw_activity_effective = True

            activity_ok_brow = True
            if (
                L["use_activity_gate"]
                and L["use_activity_gate_eyebrow"]
                and len(rest_rms_baseline_brow) >= L["activity_min_baseline_samples"]
            ):
                med = float(np.median(np.array(rest_rms_baseline_brow, dtype=float)))
                if med > 0:
                    activity_ok_brow = brow_rms >= med * L["activity_rms_multiplier_eyebrow"]

            bypass = float(L.get("brow_activity_bypass_min_conf", 0.0))
            brow_activity_effective = activity_ok_brow
            if (
                L["use_activity_gate"]
                and L["use_activity_gate_eyebrow"]
                and not activity_ok_brow
                and label == off_label
                and bypass > 0
                and conf >= bypass
            ):
                # RMS gate often stays false when forehead channel is quiet vs training; trust strong classifier.
                floor_b = confidence_floor_for_label(
                    off_label, on_label, off_label,
                    L["confidence_gate_jaw"],
                    L["confidence_gate_eyebrow"],
                )
                if conf >= floor_b and brow_margin_ok:
                    brow_activity_effective = True

            brow_tap = False
            if (
                use_dual
                and L["use_brow_tap_detector"]
                and len(rest_rms_baseline_brow) >= L["activity_min_baseline_samples"]
            ):
                med_b = float(np.median(np.array(rest_rms_baseline_brow, dtype=float)))
                p2p_b = float(np.max(x_b) - np.min(x_b))
                if med_b > 0:
                    brow_tap = (brow_rms >= med_b * L["brow_tap_rms_multiplier"]) and (
                        p2p_b >= L["brow_tap_min_p2p"]
                    )

            jaw_tap = False
            if (
                use_dual
                and L["use_jaw_tap_detector"]
                and len(rest_rms_baseline_jaw) >= L["activity_min_baseline_samples"]
            ):
                med_j = float(np.median(np.array(rest_rms_baseline_jaw, dtype=float)))
                p2p_j = float(np.max(x_j) - np.min(x_j))
                if med_j > 0:
                    jaw_tap = (jaw_rms >= med_j * L["jaw_tap_rms_multiplier"]) and (
                        p2p_j >= L["jaw_tap_min_p2p"]
                    )

            jaw_eligible: str | None = None
            brow_eligible: str | None = None
            if label == on_label and conf_ok and jaw_margin_ok and jaw_activity_effective:
                jaw_eligible = on_label
            elif (
                jaw_tap
                and jaw_activity_effective
                and (not L["jaw_tap_only_when_armed"] or armed)
            ):
                # Classifier often says rest during clench when brow channel is calm; use jaw ADC dynamics.
                jaw_eligible = on_label
            if (label == off_label and conf_ok and brow_margin_ok and brow_activity_effective) or brow_tap:
                brow_eligible = off_label

            if not armed:
                armed_jaw_accum = 0
            elif jaw_eligible == on_label:
                armed_jaw_accum += 1
            else:
                armed_jaw_accum = 0

            consensus_jaw.append(jaw_eligible)
            consensus_brow.append(brow_eligible)

            jaw_consensus = False
            if len(consensus_jaw) == L["consensus_n_jaw"]:
                need = int(L["consensus_jaw_min_votes"])
                votes = sum(1 for e in consensus_jaw if e == on_label)
                jaw_consensus = votes >= need
            need_acc = int(L["consensus_jaw_armed_accum"])
            if armed and need_acc > 0 and armed_jaw_accum >= need_acc:
                jaw_consensus = True

            brow_consensus = False
            if len(consensus_brow) == L["consensus_n_eyebrow"]:
                fb = consensus_brow[0]
                if fb is not None and all(e == fb for e in consensus_brow):
                    brow_consensus = True

            cooled = (now - last_action_at) >= L["cooldown_s"]
            eyebrow_toggle_cooled = (now - last_eyebrow_toggle_at) >= L[
                "eyebrow_toggle_cooldown_s"
            ]

            state_tag = "ARMED" if armed else "IDLE"
            if label == off_label:
                act_tag = "Y" if brow_activity_effective else "N"
            else:
                act_tag = "Y" if jaw_activity_effective else "N"
            line = (
                f"[{state_tag}] pred={label:13s} conf={conf*100:5.1f}% margin={margin*100:5.1f}% "
                f"act={act_tag}"
            )

            handled = False

            if brow_consensus:
                if not eyebrow_toggle_cooled:
                    print(line + "  -> eyebrow toggle cooldown: ignored")
                    consensus_brow.clear()
                    handled = True
                elif (now - last_eyebrow_tap_at) < L["eyebrow_tap_debounce_s"]:
                    print(line + "  -> eyebrow tap debounce: ignored")
                    consensus_brow.clear()
                    handled = True
                else:
                    last_eyebrow_tap_at = now
                    if now > eyebrow_tap_expires_at:
                        eyebrow_taps = 0
                    eyebrow_taps += 1
                    eyebrow_tap_expires_at = now + L["eyebrow_double_tap_window_s"]

                    if eyebrow_taps < 2:
                        print(
                            line
                            + f"  -> eyebrow tap 1/2 (raise again within {L['eyebrow_double_tap_window_s']:.1f}s to toggle)"
                        )
                    else:
                        eyebrow_taps = 0
                        if not armed:
                            armed = True
                            armed_until = (
                                now + L["arm_timeout_s"]
                                if L["arm_timeout_s"] > 0
                                else float("inf")
                            )
                            print(
                                tmsg(
                                    line
                                    + "  -> ARMED (double-eyebrow confirm; jaw to toggle plug)"
                                )
                            )
                        else:
                            armed = False
                            print(
                                tmsg(
                                    line
                                    + "  -> DISARMED (double-eyebrow cancel, no hardware)"
                                )
                            )
                        last_eyebrow_toggle_at = now

                    clear_consensus()
                    handled = True

            elif jaw_consensus and armed and cooled:
                queried = tuya_query_on()
                if last_known_plug_on is not None:
                    want_on = not last_known_plug_on
                elif queried is not None:
                    want_on = not queried
                else:
                    want_on = True

                action = "LIGHT_ON" if want_on else "LIGHT_OFF"
                print(tmsg(line + f"  -> ACTION: {action} (jaw while armed)"))
                last_action_at = now
                clear_consensus()
                if L["stay_armed_after_jaw"]:
                    armed = True
                    armed_until = (
                        now + L["arm_timeout_s"]
                        if L["arm_timeout_s"] > 0
                        else float("inf")
                    )
                else:
                    armed = False
                handled = True

                if tuya_ready:
                    ok = tuya_turn(want_on)
                    print(tmsg(f"  Tuya: {'OK' if ok else 'FAILED'}"))
                    if ok:
                        last_known_plug_on = want_on
                elif ifttt_key:
                    ev = ifttt_event_on if want_on else ifttt_event_off
                    ok = trigger_ifttt(ev)
                    print(tmsg(f"  IFTTT: {'OK' if ok else 'FAILED'}"))
                    if ok:
                        last_known_plug_on = want_on
                else:
                    print("  No Tuya/IFTTT: gesture recognized only (no hardware)")

            elif jaw_consensus and armed and not cooled:
                print(line + "  -> cooldown: jaw deferred (stay armed or disarm with eyebrow)")
                consensus_jaw.clear()
                armed_jaw_accum = 0
                handled = True

            elif jaw_consensus and not armed:
                print(line + "  -> jaw: ignored (not armed)")
                clear_consensus()
                handled = True

            if not handled:
                if jaw_eligible is None and brow_eligible is None:
                    print(line + "  -> gates: NO_ACTION")
                else:
                    print(line + "  -> consensus: building")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
