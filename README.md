# 🚚 Multilabel Shipment Mode Prediction App

This Streamlit web app predicts the **most suitable shipment mode(s)** (Air, Road, Rail, Sea) for products based on their features. It uses a **multi-label classification model** trained on 2000 product records with 74 features.

🔗 [Check out the live app here](https://multilabelclassification-predictingshipmentmodes-ipnnxfvsnz73h.streamlit.app)

---

## 📌 Features

- **Input form** to enter product attributes like weight, dimensions, flammability index, etc.
- **Threshold-based predictions** to view how prediction confidence changes.
- **Most confident shipment mode** at the highest threshold.
- **Interactive visualizations** comparing different models by F1-score, precision, and recall.
- **Clean UI** with sidebar and charts using Seaborn and Plotly.

---

## 🚀 How It Works

### 🔄 Multi-label Classification

Each product can be shipped by **multiple modes**. This is different from single-label classification.  
Example:
- A product might be shipped by both **Road** and **Sea**, depending on its properties.

### 📉 Thresholds & Probabilities

The model predicts a **probability** for each shipment mode.  
- For example:  
- Air: 0.81
- Road: 0.42
- Rail: 0.17
- Sea: 0.02

- A **threshold** is applied to these probabilities:
- If `probability >= threshold`, that mode is **selected**.
- Lower thresholds select more modes (more flexible).
- Higher thresholds select fewer, more **confident** modes.

- The app evaluates at:
- `Threshold 0.20`
- `Threshold 0.40`
- `Threshold 0.60`
- `Threshold 0.80`  
This gives a sense of how predictions evolve.

---
