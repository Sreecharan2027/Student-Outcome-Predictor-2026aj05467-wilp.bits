import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             classification_report, ConfusionMatrixDisplay)

st.set_page_config(page_title="Student Outcome Predictor", layout="wide")

MODEL_DIR = Path(__file__).parent / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_artifact(filename):
    return joblib.load(MODEL_DIR / filename)


def score_model(pipe, X, y_true, n_classes):
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                            labels=list(range(n_classes)))
    except ValueError:   #a class missing from the uploaded file
        auc = float("nan")
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred


#Header
st.title("Student Dropout and Success Predictor")
st.caption("Upload the held-out test CSV, pick a model, and inspect how it performs. "
           "Models are pre-trained and are not retrained on your upload.")

#Controls
with st.container(border=True):
    st.subheader("Upload and select model")
    ctrl_left, ctrl_right = st.columns([2, 1])
    with ctrl_left:
        uploaded = st.file_uploader(
            "Test data (CSV, must include the Target column)", type="csv"
        )
    with ctrl_right:
        model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

if uploaded is None:
    st.info("Upload `test_data.csv` above to begin.")
    st.stop()

#reading date
df = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")
df.columns = df.columns.str.strip()

if "Target" not in df.columns:
    st.error("The uploaded file needs a 'Target' column.")
    st.stop()

le = load_artifact("label_encoder.joblib")
unknown = set(df["Target"].unique()) - set(le.classes_)
if unknown:
    st.error(f"Unexpected labels in Target: {unknown}. Expected: {list(le.classes_)}")
    st.stop()

pipe = load_artifact(MODEL_FILES[model_name])
required = list(pipe.feature_names_in_)
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

X = df[required]
y_true = le.transform(df["Target"])
n_classes = len(le.classes_)

st.subheader("Data preview")
st.write(f"{df.shape[0]} rows, {len(required)} features")
st.dataframe(df.head(), width='stretch')

#model selection and results
metrics, y_pred = score_model(pipe, X, y_true, n_classes)

st.subheader(f"Results: {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=list(range(n_classes)),
        display_labels=le.classes_, ax=ax, cmap="Reds", colorbar=False)
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y_true, y_pred, labels=list(range(n_classes)),
                                   target_names=le.classes_, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3),width='stretch')

#final all model results
st.subheader("All models on this data")
rows = {}
for name, fname in MODEL_FILES.items():
    m, _ = score_model(load_artifact(fname), df[required], y_true, n_classes)
    rows[name] = m
st.dataframe(pd.DataFrame(rows).T.round(4), width='stretch')