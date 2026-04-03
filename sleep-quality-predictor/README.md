# Sleep Quality Predictor

Machine learning mini-project: predicts **Good / Average / Poor** sleep quality using scikit-learn, with a Streamlit UI and optional Cybernaut-style Word report.

## Setup

```bash
cd sleep-quality-predictor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

1. **Kaggle (recommended):** Download [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) and save as:

   `data/Sleep_health_and_lifestyle_dataset.csv`

2. **Demo data (no Kaggle account):**

   ```bash
   python train_model.py --generate-demo
   python train_model.py --data data/demo_sleep_data.csv
   ```

## Train

```bash
python train_model.py
```

Trains **Logistic Regression, Random Forest, Decision Tree, and SVM**, picks the best by **macro-F1**, and writes:

- `artifacts/best_model.joblib`
- `artifacts/metadata.json`

## Run the app

Run from the `sleep-quality-predictor` directory so `.streamlit/config.toml` applies (light theme, readable text):

```bash
streamlit run app.py
```

Open the URL shown in the terminal. Use **Predict Sleep Quality** for the model output and rule-based tips; **Reset Inputs** restores defaults; enable **Track history** for a session table.

## Word report (Cybernaut-style)

```bash
python generate_report.py
```

Creates **`Sleep_Quality_Predictor_Report.docx`** in this folder. To add screenshots, save PNG or JPEG files under `assets/screenshots/` and run the script again.

## Project layout

| File | Role |
|------|------|
| `train_model.py` | Load CSV, preprocess, compare models, save artifacts |
| `app.py` | Streamlit interface |
| `tips.py` | Rule-based sleep suggestions |
| `generate_report.py` | Builds the Word document |

## Notes

- The model uses dataset columns (sleep duration, activity, stress, demographics, vitals, etc.). Fields such as **screen time before bed** and **caffeine** inform **tips only**, not the classifier, so the app stays consistent with the training data.
