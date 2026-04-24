# 🌾 Crop Yield Prediction — Nepal

> Predicting district-level crop yields across Nepal using meteorological data and machine learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crop-yield-prediction-nepal-nhugtaob97vm7rnp2mj9be.streamlit.app/)

**[→ Open Live App](https://crop-yield-prediction-nepal-nhugtaob97vm7rnp2mj9be.streamlit.app/)**

---

## 📌 Overview

This project builds a full machine learning pipeline to predict crop yields (kg/ha) at the district level across Nepal. It combines satellite-derived meteorological data from NASA POWER with historical agricultural statistics from Nepal's Ministry of Agriculture and Livestock Development (MoALD).

The final model — XGBoost — is deployed as an interactive web application where users can input district, crop type, and weather parameters to get a predicted yield.

This work was **presented at ICRTAI 2025** (International Conference on Recent Trends in Artificial Intelligence).

---

## 🚀 Live Demo

🔗 [https://crop-yield-prediction-nepal-nhugtaob97vm7rnp2mj9be.streamlit.app/](https://crop-yield-prediction-nepal-nhugtaob97vm7rnp2mj9be.streamlit.app/)

---

## 📁 Repository Structure

```
crop-yield-prediction-nepal/
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md
│
├── model/
│   ├── xgb_model.json      # Trained XGBoost model
│   ├── scaler.pkl          # StandardScaler
│   └── feature_columns.pkl # Feature column names
│
└── notebooks/
    └── Final_Model.ipynb   # Full training pipeline & analysis
```

---

## 🌦️ Input Features

| Feature | Description | Source |
|--------|-------------|--------|
| `YEAR` | Year of prediction | MoALD |
| `PRECTOT` | Total precipitation (mm) | NASA POWER |
| `T2M` | Mean temperature at 2m (°C) | NASA POWER |
| `T2M_MAX` | Max temperature at 2m (°C) | NASA POWER |
| `T2M_MIN` | Min temperature at 2m (°C) | NASA POWER |
| `RH2M` | Relative humidity at 2m (%) | NASA POWER |
| `DISTRICT` | One of 75 districts of Nepal | MoALD |
| `Item` | Crop type (Paddy, Maize, Wheat, Millet) | MoALD |

---

## 🤖 Model

- **Algorithm:** XGBoost Regressor
- **Target:** Crop yield in kg/ha (log1p transformed during training)
- **Preprocessing:** One-hot encoding for categorical variables, StandardScaler for numerical features
- **Evaluation:** R² score, MSE, residual analysis

XGBoost outperformed all other tested models including Random Forest, SVR, Ridge, Lasso, and Decision Tree.

---

## 📊 Results

- Best model: **XGBoost**
- Target variable log-transformed (log1p) for better regression performance
- Predictions inverse-transformed (expm1) before display

See `notebooks/Final_Model.ipynb` for full training details, learning curves, and residual plots.

---

## 🛠️ Run Locally

```bash
git clone https://github.com/pratik-9051/crop-yield-prediction-nepal
cd crop-yield-prediction-nepal
pip install -r requirements.txt
streamlit run app.py
```

---

## 📚 Data Sources

- **MoALD** — Ministry of Agriculture and Livestock Development, Government of Nepal
- **NASA POWER** — [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/)

---

## 🎓 Research

Presented at **ICRTAI 2025** — International Conference on Recent Trends in Artificial Intelligence.

---

## 👤 Author

**Pratik Ghimire**
Agricultural Engineer | AI & Precision Agriculture Researcher
Kathmandu, Nepal

🔗 [Portfolio](https://pratik-9051.github.io) · [GitHub](https://github.com/pratik-9051)
