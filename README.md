# ❤️ CardioGuard AI: Heart Disease Risk Assessment & Prediction

An end-to-end Machine Learning project and interactive web application designed to evaluate cardiovascular disease risk based on patient lifestyle, medical history, and clinical indicators from the CDC's BRFSS 2015 dataset.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

Cardiovascular diseases (CVDs) remain the leading cause of death globally. Early diagnosis and proactive lifestyle intervention are essential for preventing heart attacks and chronic heart conditions. 

This project analyzes the **Behavioral Risk Factor Surveillance System (BRFSS) 2015** survey published by the CDC (comprising **253,680 records** across **21 health risk indicators**) and provides:
1. **In-depth Exploratory Data Analysis (EDA)** and visualization of health trends.
2. **Machine Learning Classifiers** (Logistic Regression & Random Forest) optimized for imbalanced medical data.
3. **Hyperparameter Tuning** via `GridSearchCV` and `RandomizedSearchCV`.
4. **Interactive Streamlit Web Dashboard** delivering real-time patient risk assessment, visual feedback, and preventative clinical advice.

---

## 📊 Dataset Description

* **Source:** Centers for Disease Control and Prevention (CDC) - BRFSS 2015.
* **Volume:** 253,680 survey respondents.
* **Target Variable:** `HeartDiseaseorAttack` (Binary: `0` = No Heart Disease, `1` = Reported CHD or Myocardial Infarction).
* **Class Imbalance:**
  * No Heart Disease (`0`): **90.6%** (229,787 cases)
  * Heart Disease (`1`): **9.4%** (23,893 cases)

### Key Feature Categories
* **Demographics:** Age group (1–13 scale), Biological Sex, Education level, Household income.
* **Clinical Factors:** High Blood Pressure (`HighBP`), High Cholesterol (`HighChol`), Cholesterol Check (`CholCheck`), BMI, Diabetes status (0=None, 1=Pre, 2=Diabetic), Stroke history, Difficulty walking (`DiffWalk`).
* **Behavioral & Lifestyle:** Smoking history, Heavy alcohol consumption, Physical activity in the last 30 days, Daily fruit & vegetable intake.
* **Socio-Economic & Access:** Health insurance coverage, Doctor visit cost barrier (`NoDocbcCost`).
* **Self-Reported Health:** General health rating (`GenHlth` 1=Excellent to 5=Poor), Days of poor physical health (`PhysHlth`), Days of poor mental health (`MentHlth`).

---

## 🧠 Model Training & Performance

Because healthcare datasets are heavily skewed, raw accuracy is an ineffective metric (a naive model predicting `0` always achieves 90.6% accuracy but catches 0% of disease cases). We employed **class-weighted loss (`class_weight='balanced'`)** to maximize **Recall** and **ROC-AUC**.

### Model Comparison Table

| Model | Hyperparameter Tuning Method | Best Parameters | ROC-AUC | Recall (Disease) | Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | `GridSearchCV` (3-fold CV) | `{'C': 1.0, 'solver': 'lbfgs'}` | **83.55%** | 78.00% | 74.00% |
| **Random Forest (Deployed)** | `RandomizedSearchCV` (Stratified CV) | `{'n_estimators': 150, 'max_depth': 8, 'min_samples_leaf': 7}` | 83.36% | **80.00%** | 73.00% |

> ### 💡 Why Recall over Raw Accuracy?
> In cardiovascular diagnostics, a **False Negative** (declaring an individual healthy when they are actually at risk of a cardiac event) can be life-threatening. A **False Positive** merely prompts a precautionary medical examination. By tuning our Random Forest classifier to prioritize **Recall (80.00%)**, we ensure that 4 out of 5 high-risk individuals are caught early.

---

## 📁 Repository Structure

```plaintext
heart_disease_prediction/
│
├── data/
│   └── heart_disease_health_indicators_BRFSS2015.csv   # CDC BRFSS dataset
│
├── heart_disease_eda_modeling.ipynb                   # Complete Jupyter Notebook (EDA, Modeling, Tuning)
├── app.py                                             # Streamlit Web Application
├── requirements.txt                                   # Python dependencies
├── README.md                                          # Project documentation
│
├── heart_disease_model.joblib                         # Serialized Tuned Random Forest model
├── scaler.joblib                                      # Serialized StandardScaler for continuous inputs
├── test_model.py                                      # Programmatic validation test suite
│
└── assets/                                            # Visualizations & training plots
    ├── target_distribution.png
    ├── correlation_matrix.png
    ├── health_vs_heart_disease.png
    ├── roc_curve_comparison.png
    └── feature_importance.png
```

---

## 🚀 Getting Started & Local Setup

Follow these steps to set up and run the project locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/Manish1813/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
After executing, the dashboard will automatically open in your default browser at:
`http://localhost:8501`

---

## 🔍 Key Exploratory Insights

1. **Self-Rated Health as a Strong Predictor:** Respondents reporting poor general health (`GenHlth = 5`) experienced a heart disease rate more than **5x higher** than those rating their health excellent (`GenHlth = 1`).
2. **Hypertension & Cholesterol Correlation:** More than 75% of respondents who suffered a heart attack or coronary event had prior diagnoses of High Blood Pressure and High Cholesterol.
3. **Multivariate Age Risk:** Cardiac risk escalated significantly for individuals in age brackets **7 (50–54)** and above, particularly when combined with an elevated BMI (>30).

---

## 🧪 Model Testing

A backend verification script is included to test deterministic output on simulated patient profiles:

```bash
python test_model.py
```
* **Low-Risk Profile:** ~6.25% cardiac risk probability.
* **High-Risk Profile:** ~92.62% cardiac risk probability.

---

## 📜 Disclaimer
This software and model predictions are intended strictly for **educational and research purposes**. They do not constitute certified medical diagnosis, clinical advice, or treatment plans. Consult a qualified medical practitioner for personal health assessments.

---

## 👤 Author
* **Manish Tadas** - *Data Science & Analytics Student*
* GitHub: [Manish1813](https://github.com/Manish1813)
* LinkedIn: [Manish Tadas](https://www.linkedin.com/in/manish-tadas)
