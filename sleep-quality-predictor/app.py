"""
Sleep Quality Predictor — Streamlit UI.
"""
from __future__ import annotations

import json
import os
from datetime import time

import joblib
import pandas as pd
import streamlit as st

from tips import collect_tips, input_dict_for_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "best_model.joblib")
META_PATH = os.path.join(BASE_DIR, "artifacts", "metadata.json")

CAFFEINE_OPTIONS = ["None", "Low", "Moderate", "High"]
MOOD_OPTIONS = ["Happy", "Neutral", "Sad", "Anxious"]


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
        html, body, .stApp {
            font-family: 'Poppins', 'Segoe UI', system-ui, sans-serif;
            color: #263238;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(165deg, #e8eaf6 0%, #ede7f6 35%, #f3e5f5 100%);
        }
        [data-testid="stHeader"] { background: rgba(255,255,255,0.5); }
        .block-container { padding-top: 1.2rem; max-width: 920px; }
        /* Widget labels — fix low-contrast / theme clashes */
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {
            color: #1a237e !important;
        }
        .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: #37474f;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
            color: #311b92 !important;
        }
        /* Inputs on light cards */
        [data-baseweb="input"] input, [data-baseweb="select"] > div {
            color: #212121 !important;
        }
        /* Slider, radio, checkbox */
        [data-testid="stSlider"] label { color: #1a237e !important; }
        [data-testid="stRadio"] label, [data-testid="stRadio"] p,
        [data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
            color: #1a237e !important;
        }
        [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p {
            color: #1a237e !important;
        }
        /* Expanders: readable header + content */
        [data-testid="stExpander"] details {
            background: rgba(255,255,255,0.95);
            border: 1px solid #b39ddb;
            border-radius: 12px;
        }
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {
            color: #311b92 !important;
            font-weight: 600 !important;
        }
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
            color: #263238 !important;
        }
        .sleep-hero {
            text-align: center;
            padding: 1.25rem 1rem 0.5rem;
            color: #311b92;
        }
        .sleep-hero h1 { font-weight: 700; font-size: 2rem; margin: 0; letter-spacing: -0.02em; }
        .sleep-hero p { color: #4527a0; margin: 0.35rem 0 0; font-size: 1.05rem; }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #5c6bc0, #7e57c2);
            border: none;
            font-weight: 600;
            color: #ffffff !important;
        }
        .stButton > button[kind="secondary"] {
            border-color: #7e57c2;
            color: #4527a0 !important;
            background: #ffffff !important;
        }
        [data-testid="stCaption"] { color: #546e7a !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_pipeline_and_meta():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(META_PATH):
        return None, None
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return pipe, meta


def build_feature_row(
    meta: dict,
    sleep_duration: float,
    exercise_min: int,
    stress: int,
    opt_age: float | None,
    opt_hr: float | None,
    opt_steps: float | None,
    opt_gender: str,
    opt_occupation: str,
    opt_bmi: str,
    opt_disorder: str,
    opt_sys: float | None,
    opt_dia: float | None,
) -> pd.DataFrame:
    row: dict = {}
    defaults_num = meta.get("default_numeric", {})
    defaults_cat = meta.get("default_categorical", {})
    for col in meta["feature_columns"]:
        if col == "Sleep Duration":
            row[col] = float(sleep_duration)
        elif col == "Physical Activity Level":
            row[col] = float(exercise_min)
        elif col == "Stress Level":
            row[col] = int(max(1, min(10, int(stress))))
        elif col == "Age" and opt_age is not None:
            row[col] = float(opt_age)
        elif col == "Heart Rate" and opt_hr is not None:
            row[col] = float(opt_hr)
        elif col == "Daily Steps" and opt_steps is not None:
            row[col] = float(opt_steps)
        elif col == "Gender":
            row[col] = opt_gender or defaults_cat.get(col, "Unknown")
        elif col == "Occupation":
            row[col] = opt_occupation or defaults_cat.get(col, "Unknown")
        elif col == "BMI Category":
            row[col] = opt_bmi or defaults_cat.get(col, "Unknown")
        elif col == "Sleep Disorder":
            row[col] = opt_disorder or defaults_cat.get(col, "Unknown")
        elif col == "BP_Systolic" and opt_sys is not None:
            row[col] = float(opt_sys)
        elif col == "BP_Diastolic" and opt_dia is not None:
            row[col] = float(opt_dia)
        elif col in meta.get("numeric_columns", []):
            row[col] = float(defaults_num.get(col, 0.0))
        else:
            row[col] = str(defaults_cat.get(col, "Unknown"))
    return pd.DataFrame([row])


def apply_form_defaults(meta: dict | None, clear_result: bool = True) -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    dn = (meta or {}).get("default_numeric", {})
    dc = (meta or {}).get("default_categorical", {})
    st.session_state.sleep_dur = float(dn.get("Sleep Duration", 7.0))
    st.session_state.exercise = int(dn.get("Physical Activity Level", 45))
    st.session_state.stress = int(dn.get("Stress Level", 5))
    st.session_state.screen = 30
    st.session_state.caffeine = CAFFEINE_OPTIONS[0]
    st.session_state.mood = MOOD_OPTIONS[1]
    st.session_state.interruptions = "No"
    st.session_state.bedtime = time(23, 0)
    st.session_state.wake = time(7, 0)
    st.session_state.opt_age = float(dn.get("Age", 35))
    st.session_state.opt_hr = float(dn.get("Heart Rate", 72))
    st.session_state.opt_steps = float(dn.get("Daily Steps", 6000))
    st.session_state.opt_sys = float(dn.get("BP_Systolic", 120))
    st.session_state.opt_dia = float(dn.get("BP_Diastolic", 80))
    st.session_state.opt_gender = dc.get("Gender", "Male")
    st.session_state.opt_occupation = dc.get("Occupation", "Software Engineer")
    st.session_state.opt_bmi = dc.get("BMI Category", "Normal")
    st.session_state.opt_disorder = dc.get("Sleep Disorder", "None")
    if clear_result:
        st.session_state.last_prediction = None
        st.session_state.last_probs = None
        st.session_state.last_tips = None


def main() -> None:
    st.set_page_config(
        page_title="Sleep Quality Predictor",
        page_icon="🌙",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_theme()

    pipe, meta = load_pipeline_and_meta()

    # Reset must run before any widget with bound keys is created (Streamlit restriction).
    if st.session_state.pop("pending_reset", False):
        apply_form_defaults(meta, clear_result=True)

    if "sleep_dur" not in st.session_state:
        apply_form_defaults(meta, clear_result=True)

    st.markdown(
        '<div class="sleep-hero"><h1>🌙 Sleep Quality Predictor</h1>'
        "<p>A smart way to monitor and improve your sleep</p></div>",
        unsafe_allow_html=True,
    )

    if pipe is None or meta is None:
        st.warning(
            "Trained model not found. From this folder run:\n\n"
            "`python train_model.py --generate-demo`\n\n"
            "`python train_model.py --data data/demo_sleep_data.csv`\n\n"
            "Or place **Sleep_health_and_lifestyle_dataset.csv** in `data/` and run "
            "`python train_model.py`."
        )
        return

    with st.expander("How predictions work", expanded=False):
        st.markdown(
            """
            **Model:** supervised classifier trained on lifestyle and health fields from the
            Sleep Health and Lifestyle-style dataset (Poor / Average / Good), comparing
            Logistic Regression, Random Forest, Decision Tree, and SVM — the best model by
            macro-F1 is loaded here.

            **Tips:** caffeine, screen time, mood, and interruptions are **not** inputs to
            the ML model; they drive **rule-based suggestions** alongside your prediction.
            """
        )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("##### 🛏️ Rest")
        sleep_duration = st.number_input(
            "Sleep Duration (hours)",
            min_value=0.0,
            max_value=16.0,
            step=0.25,
            key="sleep_dur",
            help="Total hours you slept last night",
        )
        bedtime = st.time_input("Bedtime", key="bedtime")
        wake = st.time_input("Wake-up Time", key="wake")
    with c2:
        st.markdown("##### ☀️ Daytime habits")
        exercise_min = st.number_input(
            "Exercise Duration (minutes)",
            min_value=0,
            max_value=300,
            step=5,
            key="exercise",
            help="Physical activity during the day",
        )
        screen_min = st.number_input(
            "Screen Time Before Bed (minutes)",
            min_value=0,
            max_value=300,
            step=5,
            key="screen",
        )
        stress = st.slider("Stress Level (0–10)", 0, 10, key="stress")

    st.markdown("##### 🧠 Evening state")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        caffeine = st.selectbox("Caffeine Intake", CAFFEINE_OPTIONS, key="caffeine")
    with cc2:
        mood = st.selectbox("Mood Before Sleep", MOOD_OPTIONS, key="mood")
    with cc3:
        interruptions = st.radio(
            "Sleep Interruptions",
            ["No", "Yes"],
            horizontal=True,
            key="interruptions",
            help="Did you wake during the night?",
        )

    with st.expander("Optional profile (matches training columns)", expanded=False):
        oc1, oc2 = st.columns(2)
        with oc1:
            opt_age = st.number_input("Age", 15, 90, key="opt_age")
            opt_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="opt_gender")
            opt_occupation = st.text_input("Occupation", key="opt_occupation")
        with oc2:
            opt_bmi = st.selectbox(
                "BMI Category",
                ["Normal", "Overweight", "Obese", "Normal Weight"],
                key="opt_bmi",
            )
            opt_disorder = st.selectbox(
                "Sleep Disorder",
                ["None", "Insomnia", "Sleep Apnea", "Unknown"],
                key="opt_disorder",
            )
        oc3, oc4, oc5 = st.columns(3)
        with oc3:
            opt_hr = st.number_input("Heart Rate (bpm)", 50, 120, key="opt_hr")
        with oc4:
            opt_steps = st.number_input("Daily Steps", 1000, 25000, step=500, key="opt_steps")
        with oc5:
            st.caption("Blood pressure")
            o5a, o5b = st.columns(2)
            with o5a:
                opt_sys = st.number_input("Systolic", 80, 200, key="opt_sys")
            with o5b:
                opt_dia = st.number_input("Diastolic", 50, 120, key="opt_dia")

    btn1, btn2, btn3 = st.columns([1.2, 1, 1])
    with btn1:
        predict = st.button("🔮 Predict Sleep Quality", type="primary", use_container_width=True)
    with btn2:
        reset = st.button("↺ Reset Inputs", use_container_width=True)
    with btn3:
        show_history = st.checkbox("Track history", value=True)

    if reset:
        st.session_state.pending_reset = True
        st.rerun()

    if predict:
        X = build_feature_row(
            meta,
            sleep_duration=float(sleep_duration),
            exercise_min=int(exercise_min),
            stress=int(stress),
            opt_age=float(st.session_state.opt_age),
            opt_hr=float(st.session_state.opt_hr),
            opt_steps=float(st.session_state.opt_steps),
            opt_gender=str(st.session_state.opt_gender),
            opt_occupation=str(st.session_state.opt_occupation),
            opt_bmi=str(st.session_state.opt_bmi),
            opt_disorder=str(st.session_state.opt_disorder),
            opt_sys=float(st.session_state.opt_sys),
            opt_dia=float(st.session_state.opt_dia),
        )
        pred = pipe.predict(X)[0]
        probs = None
        if hasattr(pipe, "predict_proba"):
            pr = pipe.predict_proba(X)[0]
            classes = list(pipe.classes_)
            probs = dict(zip(classes, [float(x) for x in pr]))
        st.session_state.last_prediction = str(pred)
        st.session_state.last_probs = probs

        tips = collect_tips(
            sleep_duration_h=float(sleep_duration),
            stress=int(stress),
            exercise_min=int(exercise_min),
            screen_min=int(screen_min),
            caffeine=str(caffeine),
            mood=str(mood),
            interruptions=str(interruptions),
            bedtime=bedtime,
            wake=wake,
            predicted_quality=str(pred),
        )
        st.session_state.last_tips = tips

        if show_history:
            entry = {
                "prediction": str(pred),
                "inputs": input_dict_for_history(
                    float(sleep_duration),
                    int(stress),
                    int(exercise_min),
                    int(screen_min),
                    str(caffeine),
                    str(mood),
                    str(interruptions),
                    bedtime,
                    wake,
                ),
            }
            st.session_state.history = [entry] + st.session_state.history[:19]

    if st.session_state.get("last_prediction"):
        st.markdown("---")
        st.markdown("### 📊 Result")
        pred = st.session_state.last_prediction
        color = {"Good": "#2e7d32", "Average": "#f9a825", "Poor": "#c62828"}.get(pred, "#4527a0")
        st.markdown(
            f'<p style="font-size:1.35rem;font-weight:600;color:{color};">Predicted Sleep Quality: {pred}</p>',
            unsafe_allow_html=True,
        )
        probs = st.session_state.get("last_probs")
        if probs:
            st.caption("Estimated class probabilities")
            st.bar_chart(
                pd.DataFrame(
                    [{"Class": k, "P": v} for k, v in sorted(probs.items(), key=lambda x: -x[1])]
                ).set_index("Class")
            )
        st.markdown("#### 💡 Suggestions")
        for t in st.session_state.get("last_tips", []):
            st.markdown(f"- {t}")

    if show_history and st.session_state.get("history"):
        st.markdown("---")
        st.markdown("### 📜 Past predictions")
        rows = []
        for i, h in enumerate(st.session_state.history):
            rows.append({"#": i + 1, "Quality": h["prediction"], **h["inputs"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(f"Loaded model: **{meta.get('best_model_name', '?')}** · {meta.get('target_binning', '')}")


if __name__ == "__main__":
    main()
