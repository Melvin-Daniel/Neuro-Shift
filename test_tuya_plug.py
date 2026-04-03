"""
Quick test: local Tuya plug ON/OFF (same LAN as laptop).
Requires: pip install tinytuya

Set env vars (PowerShell example — use YOUR values, do not commit secrets):
  $env:TUYA_DEVICE_ID="..."
  $env:TUYA_LOCAL_KEY='...'   # single quotes help with special characters
  $env:TUYA_IP="192.168.x.x"  # private IP from hotspot/router client list
  $env:TUYA_VERSION="3.3"     # optional; script also tries 3.4 / 3.5 on Err 914

Or put the same keys in .env (loaded if python-dotenv is installed).
This script loads .env with override=True so edits to .env apply even if you still
have old TUYA_* variables in the same PowerShell session.

Then: python test_tuya_plug.py

If you see "connection timed out": TUYA_IP must be the plug's private LAN address
(192.168.x.x / 10.x.x.x from your Wi‑Fi router admin), not the "ip" field from Tuya
Cloud API (that is usually your public/WAN IP). PC and plug must be on the same LAN.
Discover devices: python -m tinytuya scan   (same Wi‑Fi as the plug; may need admin)
Optional: TUYA_CONNECT_TIMEOUT=4  TUYA_CONNECT_RETRIES=2
"""
import os
import socket
import sys
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        # override=True: .env wins over stale $env:TUYA_* from an earlier session.
        # (live_emg_demo.py uses override=False so intentional shell tests still work.)
        load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    except ImportError:
        pass


def _status_ok(st: object) -> bool:
    if not isinstance(st, dict):
        return False
    return st.get("Error") is None and st.get("Err") is None


def main() -> int:
    _load_dotenv()
    did = os.environ.get("TUYA_DEVICE_ID", "").strip()
    key = os.environ.get("TUYA_LOCAL_KEY", "").strip()
    ip = os.environ.get("TUYA_IP", "").strip()
    ver = os.environ.get("TUYA_VERSION", "3.3").strip() or "3.3"
    if not (did and key and ip):
        print("Set TUYA_DEVICE_ID, TUYA_LOCAL_KEY, TUYA_IP", file=sys.stderr)
        return 2
    if len(key) != 16:
        print(
            f"WARNING: local key length is {len(key)}; Tuya expects 16. "
            "Err 914 until you re-copy the full Local Key from Tuya IoT → Cloud → Devices.",
            file=sys.stderr,
        )
    try:
        import tinytuya
    except ImportError:
        print("pip install tinytuya", file=sys.stderr)
        return 2

    try:
        timeout_s = float(os.environ.get("TUYA_CONNECT_TIMEOUT", "4") or "4")
    except ValueError:
        timeout_s = 4.0
    try:
        retry_lim = int(os.environ.get("TUYA_CONNECT_RETRIES", "2") or "2")
    except ValueError:
        retry_lim = 2
    retry_lim = max(1, min(retry_lim, 10))

    def versions_to_try() -> list[float]:
        out: list[float] = []
        for v in (float(ver), 3.3, 3.4, 3.5):
            if not any(abs(v - o) < 1e-9 for o in out):
                out.append(v)
        return out

    def make_device(pv: float) -> object:
        d = tinytuya.OutletDevice(
            did, ip, key, connection_timeout=timeout_s, connection_retry_limit=retry_lim
        )
        d.set_version(pv)
        d.set_socketTimeout(timeout_s)
        d.set_socketRetryLimit(retry_lim)
        return d

    connect_fail_hint = (
        "\n  No TCP reply from the plug. Fix:\n"
        "  - Set TUYA_IP to this device's Wi‑Fi IP (router DHCP / client list), "
        "not the Tuya Cloud JSON \"ip\" (often a public address).\n"
        "  - Use the same Wi‑Fi/LAN as the plug (not mobile data / guest Wi‑Fi isolation).\n"
        "  - Try: python -m tinytuya scan\n"
    )

    d = None
    used_ver: float | None = None
    last_st: object = None
    for pv in versions_to_try():
        d = make_device(pv)
        try:
            st = d.status()
        except (TimeoutError, socket.timeout, OSError, ConnectionRefusedError) as e:
            print(
                f"Connect failed to {ip}:6668 (protocol {pv}): {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            print(connect_fail_hint, file=sys.stderr)
            return 1
        last_st = st
        if _status_ok(st):
            used_ver = pv
            print(f"OK with protocol {pv}: status:", st)
            break
        err = st.get("Err") if isinstance(st, dict) else None
        if str(err) != "914":
            print(f"status (protocol {pv}):", st)
            return 1
    else:
        print("status: all protocol tries failed (likely wrong key / id / LAN IP)", file=sys.stderr)
        print("last response:", last_st, file=sys.stderr)
        return 1

    assert d is not None and used_ver is not None
    print("turn_on:", d.turn_on())
    print("turn_off:", d.turn_off())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
