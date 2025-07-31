import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# --- Page Config ---
st.set_page_config(page_title="Shipment Mode Prediction", layout="wide")

# --- Load Assets ---
try:
    model = joblib.load("best_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'best_model.pkl' not found.")
    st.stop()

if os.path.exists("feature_list.pkl"):
    feature_names = joblib.load("feature_list.pkl")
else:
    st.error("❌ Missing 'feature_list.pkl'. Please ensure it is in the same directory.")
    st.stop()

# --- Shipment Mode Labels ---
class_names = ["Air", "Road", "Rail", "Sea"]

# --- Optional Assets ---
metrics_df = pd.read_csv("model_comparison_metrics.csv") if os.path.exists("model_comparison_metrics.csv") else None
scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None

# --- App Title ---
st.markdown("<h1 style='text-align: center;'>🚚 Multilabel Shipment Mode Prediction App</h1>", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.markdown("## 🧾 Enter Product Features")
input_data = {}
for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0, step=0.1)

X_input = pd.DataFrame([input_data])

# --- Predict Button ---
if st.sidebar.button("🔍 Predict Shipment Modes"):
    if scaler:
        X_scaled = scaler.transform(X_input)
    else:
        X_scaled = X_input

    # --- Prediction Probabilities ---
    try:
        probs_raw = model.predict_proba(X_scaled)

        if isinstance(probs_raw, list):
            # List of arrays from OneVsRestClassifier or similar
            probs = np.array([p[0][1] if isinstance(p[0], (tuple, list, np.ndarray)) else p[0] for p in probs_raw])
            probs = np.array(probs).reshape(1, -1)
        else:
            probs = np.array(probs_raw).reshape(1, -1)
    except AttributeError:
        probs = model.predict(X_scaled).astype(float)

    # --- Only Use Shipment Modes ---
    shipment_mode_indices = [0, 1, 2, 3]
    probs_selected = probs[:, shipment_mode_indices]

    # --- Dynamic Threshold Selector ---
    st.markdown("## 🎯 Predictions Based on Adjustable Threshold")
    fixed_threshold = st.slider("Set threshold for class prediction:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    st.info(f"Using a threshold of **{fixed_threshold}** to determine shipment modes.")

    pred = (probs_selected >= fixed_threshold).astype(int)
    labels = np.nonzero(pred[0])[0]
    predicted = [class_names[i] for i in labels] if labels.any() else ["None"]
    st.success(f"**Predicted Shipment Mode(s):** {', '.join(predicted)}")

    # --- Show Probabilities ---
    st.markdown("## 📊 Predicted Probabilities for Each Mode")
    prob_df = pd.DataFrame(probs_selected, columns=class_names)
    st.dataframe(prob_df.style.format("{:.2f}"))

    # --- Plot Probabilities ---
    st.markdown("### 📈 Confidence Levels (Bar Chart)")
    fig, ax = plt.subplots()
    sns.barplot(x=class_names, y=probs_selected[0], ax=ax, palette="Blues_d")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Confidence per Shipment Mode")
    st.pyplot(fig)

    # --- Top-2 Prediction (regardless of threshold) ---
    top_n = 2
    top_indices = np.argsort(probs_selected[0])[::-1][:top_n]
    top_preds = [(class_names[i], probs_selected[0][i]) for i in top_indices]

    st.markdown(f"### 🏆 Top-{top_n} Predicted Shipment Modes (Irrespective of Threshold)")
    for mode, score in top_preds:
        st.write(f"- **{mode}** → {score:.2%} confidence")

    # --- Most Confident Prediction ---
    max_prob = probs_selected[0].max()
    top_label = class_names[np.argmax(probs_selected[0])]
    st.markdown(f"### 🔎 Most Confident Prediction: **{top_label}** with {max_prob:.2%} confidence")

# --- Divider ---
st.markdown("---")

# --- Static Evaluation Charts ---
if metrics_df is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Model Evaluation Summary")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), height=250)

    with col2:
        st.subheader("📈 Model F1-Score Comparison")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Model', y='F1-Score', data=metrics_df, palette='coolwarm', ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        st.subheader("📉 Precision, Recall, and F1 Score Radar Chart")
        melted_df = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        fig2 = px.line_polar(
            melted_df,
            r='Score', theta='Metric', color='Model', line_close=True,
            template='plotly_dark', height=500
        )
        st.plotly_chart(fig2)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    👨‍💻 **Developed by [Adarsh Agrawal](https://www.linkedin.com/in/adarsh-agrawal-3b0a76268/)**  
    """,
    unsafe_allow_html=True
)
