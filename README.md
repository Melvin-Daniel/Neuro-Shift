# Neuro-Shift

**Repository:** [github.com/Melvin-Daniel/Neuro-Shift](https://github.com/Melvin-Daniel/Neuro-Shift) · **Team:** Puranaanooru (OSSome Hacks 3.0)

**Prototype submission (form):** [Google Form](https://forms.gle/Z8jeVGrKX4qBEuxGA) — submit **GitHub link**, **optional `.env`** (not in repo), **optional ZIP** (evaluator notes), **deployed link** only if applicable. Do **not** commit `.env`, `node_modules`, `.build`, or `__pycache__`.

---

**EMG-based gesture control** for accessibility and smart-home actions: two **AD8232** muscle sensors (jaw + forehead) stream through an **Arduino**, a **Python** pipeline classifies gestures with a **Random Forest**, and the **live demo** can toggle a **Tuya** smart plug (or **IFTTT**) after a deliberate **double-eyebrow “arm”** and **jaw clench** command.

---

## Quick start (judges / demo without retraining)

1. **Clone** this repo (do not expect `.env` in the repo).
2. **Python 3.10+**, create a venv, then `pip install -r requirements.txt` from the repo root.
3. Copy **`.env.example`** → **`.env`** and fill **Tuya** fields (or leave empty to run gesture-only). For local Tuya control you need **device id**, **16-character local key**, and the plug’s **LAN IP** (e.g. `192.168.x.x`), not the public IP from Tuya Cloud.
4. Set **`emg_config.json`** → `serial.port` to your Arduino port (e.g. `COM4`), or set `EMG_SERIAL_PORT`.
5. **`emg_rf_model.pkl`** — generate with **`train_emg_model.py`** using `data/<user>/` CSVs (see [Train the model](#train-the-model)), then keep the file in the project root. **Commit the `.pkl`** to GitHub if you want the jury to run the live demo without training.
6. **Close Arduino Serial Monitor**, then run:
   - `python test_tuya_plug.py` — optional; confirms Tuya on your LAN.
   - `python live_emg_demo.py` — live EMG + armed flow + plug toggle when configured.

**Demo flow:** double **eyebrow** to arm → **jaw clench** toggles output → double eyebrow disarms.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `sketch_oct3a/` | Arduino firmware: `millis,jaw,brow` CSV lines @ 9600 baud |
| `collect_data.py` | Record labeled CSVs into `data/<user>/` |
| `train_emg_model.py` | Train `emg_rf_model.pkl` from those CSVs |
| `live_emg_demo.py` | Real-time classification + armed flow + optional Tuya/IFTTT |
| `emg_config.json` + `emg_config.py` | Serial port, model path, live thresholds |
| `backend-api/` | Optional **FastAPI** service + synthetic “EEG” command demo |
| `neural-decoder/`, `signal-simulator/` | Support code for the API demo |
| `mobile/`, `sleep-quality-predictor/` | Additional experiments (optional) |

---

## Hardware setup

1. **Two AD8232** modules: **jaw** (e.g. masseter) → **A0**, **forehead** (e.g. frontalis) → **A1** on the Arduino.
2. Upload `sketch_oct3a/sketch_oct3a.ino` (default **9600 baud**, ~10 ms between samples).
3. Connect USB; note the serial port (Windows: **COM4**, etc.).

---

## Software prerequisites

- **Python 3.10+**
- **Arduino IDE** (or equivalent) to flash the sketch

---

## Install (EMG pipeline)

From the repository root:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configure serial port and model

Edit **`emg_config.json`**:

- `serial.port` — your Arduino COM port (e.g. `COM4`, or `/dev/ttyUSB0`).
- `model.path` — usually `emg_rf_model.pkl`.

Optional environment overrides (no secrets):

- `EMG_SERIAL_PORT` — overrides JSON port.
- `EMG_MODEL_PATH` — overrides JSON model path.

**Only one app may open the serial port at a time** (close Arduino Serial Monitor before running Python). See `COM_SERIAL_NOTE.txt`.

---

## Secrets (Tuya / IFTTT)

Do **not** commit secrets. Copy **`.env.example`** to **`.env`**, fill in values, and use it locally. **`live_emg_demo.py` loads `.env` automatically** (via `python-dotenv`) from the project folder. For the hackathon **upload `.env` separately in the Google Form** as required.

| Variable | Used for |
|----------|----------|
| `TUYA_DEVICE_ID`, `TUYA_LOCAL_KEY`, `TUYA_IP`, `TUYA_VERSION` | Local Tuya control in `live_emg_demo.py` |
| `TUYA_DEVICE_TYPE` | `outlet` (default) or `bulb` for smart lights |
| `TUYA_SWITCH_DPS` | Switch data point: many plugs use `1`, many bulbs use `20` (ignored when `TYPE=bulb`, auto-detected) |
| `TUYA_PERSIST`, `TUYA_SEND_WAIT`, `TUYA_VERBOSE` | Optional tuning; see `.env.example` |
| `IFTTT_KEY`, `IFTTT_EVENT_ON`, `IFTTT_EVENT_OFF` | Optional webhook fallback if Tuya is not used |

Gesture recognition and logging work **without** Tuya/IFTTT; hardware lines simply won’t fire.

---

## Collect training data

1. Close any other program using the COM port.
2. Run:

   ```bash
   python collect_data.py
   ```

3. Enter a **user folder** (e.g. `vijay`) → files go under `data/vijay/`.
4. Enter gesture name: **`rest`**, **`eyebrow_raise`**, **`jaw_clench`** (exact names).
5. Enter duration in seconds; perform the gesture when recording starts.

**Tips:** For `rest`, include normal face motion, talking, and light chewing so the model learns “not a command.” Keep electrode placement consistent across sessions.

---

## Train the model

Requires **`data/<user>/rest.csv`**, **`jaw_clench.csv`**, **`eyebrow_raise.csv`** (dual columns: `timestamp,signal_jaw,signal_brow`).

Train **one user** at a time if folders differ in format:

**Windows (PowerShell):**

```powershell
$env:EMG_ONLY_USER="vijay"
python train_emg_model.py
```

**macOS / Linux:**

```bash
export EMG_ONLY_USER=vijay
python train_emg_model.py
```

This writes **`emg_rf_model.pkl`** in the project root. Commit it if judges should run the demo without retraining (keep repo size limits in mind).

---

## Run the live demo

```bash
python live_emg_demo.py
```

**Intended flow**

1. **Double eyebrow** (two confirmed taps within the configured window) → **ARMED** (safe to accept jaw command).
2. While **ARMED**, **jaw clench** → toggles smart plug (Tuya) or IFTTT if configured.
3. **Double eyebrow** again → **DISARM** (cancel).
4. Jaw while **IDLE** does nothing (accidental clenches ignored).

Tune gates in **`emg_config.json`** under `live` if needed (confidence, consensus windows, tap detectors, `brow_activity_bypass_min_conf`, etc.).

Stop with **Ctrl+C**.

---

## Optional: FastAPI + synthetic command test

Terminal 1 — API (from repo root):

```bash
cd backend-api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 — synthetic client (from repo root, with venv active):

```bash
pip install requests
python test_complete_system.py
```

This exercises the **brain-command classifier API** path (separate from the **EMG** live demo).

---

## What not to push to GitHub

Per submission rules, **do not** commit:

- `node_modules/`
- `.build/` or other build artifacts
- `.env` (use `.env.example` only in the repo)
- `__pycache__/` (handled by `.gitignore`)

---

## Hackathon form upload (separate `.env` + ZIP fields)

Many forms have **two** uploads: **`.env` file`** and **`Zip file`**.

1. **`.env` field** — Upload the file from your machine:  
   `C:\Users\MELVIN\neuro-shift\.env`  
   (Evaluators use this to run Tuya locally; never commit it to GitHub.)

2. **Zip field** — Build evaluator notes (no secrets inside): from repo root run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_google_form_zip.ps1
```

That creates **`neuro-shift-google-form-upload.zip`** with **`EVALUATOR_QUICKSTART.txt`** + **`FORM_UPLOAD_README.txt`**. Upload that ZIP in the zip field.

If a form only allows **one** zip and no separate `.env` field, run:  
`powershell -ExecutionPolicy Bypass -File scripts\make_google_form_zip.ps1 -IncludeEnv`

See **`WHAT_TO_UPLOAD_FOR_FORM.txt`** in the repo for a short checklist. **Do not** commit the ZIP or `.env` to GitHub.

---

## Deployed link

Not required for this prototype (hardware + local Python). If you add a hosted dashboard later, put the URL in the form.

---

## Team / contact

*(Add your team name, track, and Discord or email here.)*
