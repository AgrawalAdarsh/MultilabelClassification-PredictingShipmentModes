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

This is a **multi-label classification** problem where a product can have multiple valid shipment modes (e.g., both Road and Sea).  
The model predicts **probabilities** for each mode and uses a **threshold** to decide which ones to select.  
Lower thresholds show more options; higher ones pick only high-confidence modes.  
You can adjust the threshold using the slider in the sidebar.
**Note:** Since the dataset is skewed toward **Road**, predictions often favor it.
