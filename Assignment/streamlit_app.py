import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from ml_utils import compute_metrics, METRIC_COLS

st.set_page_config(page_title="Classifier Evaluation Lab",layout="wide")


@st.cache_resource
def load_assets():
    meta = json.load(open("model/meta.json"))
    models = {n: joblib.load(f"model/{f}") for n, f in meta["models"].items()}
    return meta, models


meta, models = load_assets()
TARGET, CLASSES, FEATS = meta["target_col"], meta["classes"], meta["feature_cols"]
cls_idx = {c: i for i, c in enumerate(CLASSES)}

#sidebar
st.sidebar.title("Controls")
upload = st.sidebar.file_uploader("1️⃣ Upload TEST data (CSV)", type="csv")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=upload is None)
choice = st.sidebar.selectbox("2️⃣ Select model", list(models.keys()))
st.sidebar.download_button("⬇️ Download sample test CSV",
                           open("test_data.csv", "rb"), "test_data.csv", "text/csv")

st.title("Classifier Evaluation Lab")

if upload is not None:
    df = pd.read_csv(upload)
elif use_sample:
    df = pd.read_csv("test_data.csv")
else:
    st.info("Upload a CSV or tick the sample-data box in the sidebar.")
    st.stop()

missing = [c for c in FEATS + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

df = df.dropna(subset=[TARGET])
y_true = df[TARGET].astype(str).map(cls_idx)
if y_true.isna().any():
    st.error(f"Target contains labels unseen in training. Expected: {CLASSES}")
    st.stop()
y_true = y_true.astype(int).to_numpy()
X = df[FEATS]


def evaluate(name):
    pipe = models[name]
    proba = pipe.predict_proba(X)
    pred = proba.argmax(axis=1)
    return pred, proba, compute_metrics(y_true, pred, proba)


tab1, tab2, tab3 = st.tabs(["Selected model", "Compare all models", "Data preview"])

with tab1:
    pred, proba, m = evaluate(choice)
    st.subheader(f"{choice}")
    cols = st.columns(6)
    for c, k in zip(cols, METRIC_COLS):
        c.metric(k, "n/a" if np.isnan(m[k]) else f"{m[k]:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(confusion_matrix(y_true, pred, labels=range(len(CLASSES))), annot=True, fmt="d",
                    cmap="crest", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y_true, pred, labels=range(len(CLASSES)),
                                    target_names=CLASSES, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

    if len(CLASSES) == 2 and len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
        ax2.plot(fpr, tpr, label=f"AUC = {m['AUC']:.3f}"); ax2.plot([0, 1], [0, 1], "k--")
        ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate"); ax2.legend()
        st.markdown("**ROC curve**"); st.pyplot(fig2)

with tab2:
    table = pd.DataFrame({n: evaluate(n)[2] for n in models}).T[METRIC_COLS]
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#2e7d32"),
                 width="stretch")
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])
    st.success(f"Best by MCC on this file: **{table['MCC'].idxmax()}**")

with tab3:
    st.write(f"{len(df)} rows × {df.shape[1]} columns")
    st.dataframe(df.head(50), width="stretch")
