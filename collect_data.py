"""
Record EMG streams for train_emg_model.py (window_size=100 samples, same Arduino sketch as live demo).

Expects sketch_oct3a.ino with two AD8232: timestamp,signal_jaw,signal_brow per line.

Saves under data/<user>/ when you enter a user name, else project root (legacy).

Serial port and baud: see emg_config.json (override with EMG_SERIAL_PORT).

Recording guidelines:
- rest: talking, chewing, idle face motion — not only perfectly still.
- eyebrow_raise / jaw_clench: deliberate repeats; same electrode placement every session.
"""
import os
import re
import serial
import time
from serial.serialutil import SerialException

from emg_config import load_emg_config

_LINE_OK = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*$")

_cfg = load_emg_config()
PORT = _cfg["serial"]["port"]
BAUD = _cfg["serial"]["baud"]

try:
    arduino = serial.Serial(PORT, BAUD, timeout=1)
except (PermissionError, SerialException) as e:
    print(f"Error: could not open port {PORT!r} @ {BAUD} baud.")
    print(f"  {e}")
    print("Fix:")
    print("  - Close Arduino IDE Serial Monitor / Serial Plotter (they lock the port).")
    print("  - Stop any other Python script using that port (e.g. live_emg_demo.py).")
    print("  - Unplug/replug the Arduino USB, then try again.")
    print(f"  - Or set EMG_SERIAL_PORT or edit emg_config.json serial.port")
    raise SystemExit(1)

time.sleep(2)

user_folder = input(
    "User folder under data/ (e.g. melvin, friend) — Enter for project root: "
).strip()

gesture_name = input("Enter gesture name (e.g., eyebrow_raise, jaw_clench, rest): ")
duration = int(input("Enter duration in seconds (e.g., 60): "))

if user_folder:
    out_dir = os.path.join("data", user_folder)
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{gesture_name}.csv")
else:
    filename = f"{gesture_name}.csv"

print(f"\nCollecting to {filename} for {duration} seconds...")
print("Start performing your gesture NOW!\n")

written = 0
skipped = 0

with open(filename, "w", encoding="utf-8", newline="\n") as f:
    f.write("timestamp,signal_jaw,signal_brow\n")

    start_time = time.time()
    while (time.time() - start_time) < duration:
        if arduino.in_waiting > 0:
            raw = arduino.readline()
            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                skipped += 1
                continue
            m = _LINE_OK.match(line)
            if not m:
                skipped += 1
                continue
            clean = f"{m.group(1)},{m.group(2)},{m.group(3)}"
            print(clean)
            f.write(clean + "\n")
            written += 1
            f.flush()

print(f"\nData collection complete. {written} samples written, {skipped} lines skipped.")
if written == 0:
    print(
        "Warning: no valid lines. Check COM port, baud (9600), USB cable, and sketch output."
    )
arduino.close()
