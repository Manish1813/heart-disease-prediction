import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="CardioGuard | Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global font styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Container styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .header-container h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        color: white !important;
    }
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.9;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Metric/Card Styling */
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1e3c72;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Risk output styling */
    .risk-high {
        background-color: #ffeef0;
        border: 2px solid #ff4d4f;
        padding: 2rem;
        border-radius: 12px;
        color: #cf1322;
        text-align: center;
        margin-top: 1.5rem;
    }
    .risk-low {
        background-color: #f6ffed;
        border: 2px solid #52c41a;
        padding: 2rem;
        border-radius: 12px;
        color: #389e0d;
        text-align: center;
        margin-top: 1.5rem;
    }
    .risk-medium {
        background-color: #fffbe6;
        border: 2px solid #faad14;
        padding: 2rem;
        border-radius: 12px;
        color: #d46b08;
        text-align: center;
        margin-top: 1.5rem;
    }
    
    /* Custom button styling */
    div.stButton > button {
        background: linear-gradient(135deg, #ff4e50 0%, #f9d423 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 30px;
        box-shadow: 0 4px 10px rgba(255, 78, 80, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 78, 80, 0.4);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- LOAD MODELS AND METADATA -----------------
@st.cache_resource
def load_models():
    model = joblib.load('heart_disease_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model or scaler: {e}. Please ensure you run the Jupyter Notebook first to generate files.")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-health.png", width=90)
    st.title("CardioGuard Engine")
    st.markdown("""
    This application utilizes a **Random Forest Classifier** trained on the CDC's **BRFSS 2015 dataset** (253,680 records).
    
    **Model Performance Metrics:**
    - **ROC-AUC:** ~80%
    - **Sensitivity (Recall):** ~78% (Optimized to identify true positive cases)
    - **Training Strategy:** Imbalanced learning using Class Weights.
    """)
    st.write("---")
    st.info("💡 **Instructions:** Provide the health indicators, habits, and demographic info in the tabs on the right, then click **'Analyze Risk Score'**.")

# ----------------- APP BODY -----------------
# Premium Header
st.markdown("""
<div class="header-container">
    <h1>❤️ CardioGuard AI: Heart Disease Risk Assessment</h1>
    <p>Predict your probability of coronary heart disease or myocardial infarction using machine learning models trained on nationwide CDC healthcare survey data.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ ML Model and Scaler files were not found. Please train the model in the notebook first to generate `heart_disease_model.joblib` and `scaler.joblib`.")
else:
    # Set up columns/tabs
    tab_predict, tab_insights, tab_info = st.tabs(["🔍 Risk Prediction Tool", "📊 Exploratory Insights", "📖 About BRFSS Dataset"])
    
    with tab_predict:
        st.subheader("Interactive Health Profiler")
        st.write("Please fill out the details in the sections below to estimate your heart disease risk status.")
        
        # Grid layout for inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 👤 Demographics & Stats")
            
            sex_choice = st.selectbox("Biological Sex", ["Female", "Male"])
            sex = 1 if sex_choice == "Male" else 0
            
            age_ranges = {
                "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, 
                "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, 
                "70-74": 11, "75-79": 12, "80 or older": 13
            }
            age_choice = st.selectbox("Age Group", list(age_ranges.keys()), index=7) # Default 55-59
            age = age_ranges[age_choice]
            
            bmi = st.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=60.0, value=25.0, step=0.1, help="BMI is calculated as weight in kg divided by height in meters squared.")
            
            # Show BMI classification
            if bmi < 18.5:
                st.caption("Classification: **Underweight** 🟡")
            elif 18.5 <= bmi < 25:
                st.caption("Classification: **Normal Weight** 🟢")
            elif 25 <= bmi < 30:
                st.caption("Classification: **Overweight** 🟡")
            else:
                st.caption("Classification: **Obese** 🔴")
                
            education_choices = {
                "Never attended school or only kindergarten": 1,
                "Grades 1 through 8 (Elementary)": 2,
                "Grades 9 through 11 (Some High School)": 3,
                "Grade 12 or GED (High School Graduate)": 4,
                "College 1 to 3 years (Some College)": 5,
                "College 4 years or more (College Graduate)": 6
            }
            edu_choice = st.selectbox("Highest Education Level", list(education_choices.keys()), index=5)
            education = education_choices[edu_choice]
            
            income_choices = {
                "Less than $10,000": 1,
                "$10,000 to less than $15,000": 2,
                "$15,000 to less than $20,000": 3,
                "$20,000 to less than $25,000": 4,
                "$25,000 to less than $35,000": 5,
                "$35,000 to less than $50,000": 6,
                "$50,000 to less than $75,000": 7,
                "$75,000 or more": 8
            }
            inc_choice = st.selectbox("Annual Household Income", list(income_choices.keys()), index=7)
            income = income_choices[inc_choice]
            
        with col2:
            st.markdown("### 🩺 Clinical Conditions")
            
            high_bp = 1 if st.selectbox("High Blood Pressure (High BP)", ["No", "Yes"]) == "Yes" else 0
            high_chol = 1 if st.selectbox("High Cholesterol", ["No", "Yes"]) == "Yes" else 0
            chol_check = 1 if st.selectbox("Cholesterol Checked in Last 5 Years?", ["Yes", "No"]) == "Yes" else 0
            
            diabetes_choice = st.selectbox("Diabetic Status", ["No Diabetes", "Prediabetes", "Diabetic"])
            diabetes = 0 if diabetes_choice == "No Diabetes" else (1 if diabetes_choice == "Prediabetes" else 2)
            
            stroke = 1 if st.selectbox("Ever Had a Stroke?", ["No", "Yes"]) == "Yes" else 0
            diff_walk = 1 if st.selectbox("Serious Difficulty Walking or Climbing Stairs?", ["No", "Yes"]) == "Yes" else 0
            
            gen_hlth_choices = {
                "Excellent": 1,
                "Very Good": 2,
                "Good": 3,
                "Fair": 4,
                "Poor": 5
            }
            gen_hlth_choice = st.selectbox("General Health Assessment", list(gen_hlth_choices.keys()), index=1)
            gen_hlth = gen_hlth_choices[gen_hlth_choice]

        with col3:
            st.markdown("### 🥗 Lifestyle & Access")
            
            smoker = 1 if st.selectbox("Smoked >100 Cigarettes in Life?", ["No", "Yes"]) == "Yes" else 0
            hvy_alcohol = 1 if st.selectbox("Heavy Drinker?", ["No", "Yes"], help="Men: >14 drinks/week. Women: >7 drinks/week.") == "Yes" else 0
            phys_activity = 1 if st.selectbox("Physical Activity in Last 30 Days? (excluding work)", ["Yes", "No"]) == "Yes" else 0
            fruits = 1 if st.selectbox("Consume Fruits 1+ Times Per Day?", ["Yes", "No"]) == "Yes" else 0
            veggies = 1 if st.selectbox("Consume Vegetables 1+ Times Per Day?", ["Yes", "No"]) == "Yes" else 0
            
            any_healthcare = 1 if st.selectbox("Have Health Care Insurance/Coverage?", ["Yes", "No"]) == "Yes" else 0
            no_doc_cost = 1 if st.selectbox("Couldn't See Doctor Due to Cost in Last 12 Months?", ["No", "Yes"]) == "Yes" else 0
            
            phys_hlth = st.slider("Days of Bad Physical Health (Last 30 Days)", 0, 30, 0)
            ment_hlth = st.slider("Days of Bad Mental Health (Last 30 Days)", 0, 30, 0)
            
        st.write("---")
        
        # Trigger prediction
        predict_btn = st.button("🔴 Analyze Risk Score")
        
        if predict_btn:
            # 1. Organize input vector (order must match original dataframe columns)
            # The columns in training set:
            # ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes', 
            #  'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 
            #  'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 
            #  'Age', 'Education', 'Income']
            
            input_dict = {
                'HighBP': high_bp,
                'HighChol': high_chol,
                'CholCheck': chol_check,
                'BMI': bmi,
                'Smoker': smoker,
                'Stroke': stroke,
                'Diabetes': diabetes,
                'PhysActivity': phys_activity,
                'Fruits': fruits,
                'Veggies': veggies,
                'HvyAlcoholConsump': hvy_alcohol,
                'AnyHealthcare': any_healthcare,
                'NoDocbcCost': no_doc_cost,
                'GenHlth': gen_hlth,
                'MentHlth': ment_hlth,
                'PhysHlth': phys_hlth,
                'DiffWalk': diff_walk,
                'Sex': sex,
                'Age': age,
                'Education': education,
                'Income': income
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # 2. Scale continuous inputs
            scaled_vals = scaler.transform(input_df[['BMI', 'MentHlth', 'PhysHlth']])
            input_df_scaled = input_df.copy()
            input_df_scaled[['BMI', 'MentHlth', 'PhysHlth']] = scaled_vals
            
            # 3. Model Prediction
            prob = model.predict_proba(input_df_scaled)[0][1] # Probability of Heart Disease (class 1)
            pred = model.predict(input_df_scaled)[0]
            
            # 4. Display result with gorgeous UI card
            st.markdown("## 📊 Clinical Diagnosis Report")
            
            # Determine Risk category
            # A threshold of 50% is standard binary threshold. But with RF class weights,
            # we can break this down: <30% Low Risk, 30% - 60% Moderate Risk, >60% High Risk.
            if prob < 0.35:
                risk_class = "risk-low"
                status = "LOW RISK"
                advice = "Your profile shows low statistical risk of cardiovascular events relative to this dataset. Continue maintaining a healthy diet, regular exercise, and standard health checkups."
            elif 0.35 <= prob < 0.65:
                risk_class = "risk-medium"
                status = "MODERATE RISK"
                advice = "You have a moderate statistical risk. Consider incorporating more cardiovascular activities, managing stress levels, checking blood pressure regularly, and discussing these results with a healthcare professional."
            else:
                risk_class = "risk-high"
                status = "HIGH RISK INDICATOR"
                advice = "Our clinical model identifies key indicators associated with elevated risk of heart disease or stroke. We highly recommend scheduling an appointment with your primary care provider for a cardiovascular screening."
                
            st.markdown(f"""
            <div class="{risk_class}">
                <h3>{status}</h3>
                <h1 style='font-size: 3.5rem; margin: 0.5rem 0;'>{prob*100:.1f}%</h1>
                <p>Calculated Risk Probability</p>
                <p style="font-size: 1.1rem; max-width: 700px; margin: 1rem auto 0 auto; color: inherit; line-height: 1.5;">
                    <strong>Recommendation:</strong> {advice}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display risk metrics
            st.write("### Profile Highlights impacting risk:")
            col_metric_1, col_metric_2, col_metric_3, col_metric_4 = st.columns(4)
            with col_metric_1:
                st.markdown(f"<div class='metric-card'><strong>Hypertension:</strong> {'🔴 Present' if high_bp == 1 else '🟢 Normal'}</div>", unsafe_allow_html=True)
            with col_metric_2:
                st.markdown(f"<div class='metric-card'><strong>Cholesterol:</strong> {'🔴 High' if high_chol == 1 else '🟢 Normal'}</div>", unsafe_allow_html=True)
            with col_metric_3:
                st.markdown(f"<div class='metric-card'><strong>Diabetes status:</strong> {'🔴 Yes' if diabetes == 2 else ('🟡 Prediabetes' if diabetes == 1 else '🟢 None')}</div>", unsafe_allow_html=True)
            with col_metric_4:
                st.markdown(f"<div class='metric-card'><strong>General Health:</strong> Rating {gen_hlth}/5</div>", unsafe_allow_html=True)

    with tab_insights:
        st.subheader("Model & Data Insights")
        st.write("Below are the exploratory graphs generated during model training. These explain the correlations and features driving our heart disease predictions.")
        
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("#### 1. Target Class Imbalance")
            if os.path.exists("target_distribution.png"):
                st.image("target_distribution.png", caption="CDC Survey respondents imbalance (9.4% have heart disease or attack). This highlights why standard accuracy is misleading and class weighting is required.")
            else:
                st.info("Run training notebook to generate distribution plot.")
                
            st.markdown("#### 3. Feature Importance (Random Forest)")
            if os.path.exists("feature_importance.png"):
                st.image("feature_importance.png", caption="The top features that determine risk score inside the Random Forest model. General health (GenHlth), High Blood Pressure (HighBP), and Age are key risk factors.")
            else:
                st.info("Run training notebook to generate importance plot.")
                
        with col_img2:
            st.markdown("#### 2. Risk Rate by General Health")
            if os.path.exists("health_vs_heart_disease.png"):
                st.image("health_vs_heart_disease.png", caption="Observe the exponential rise in actual heart disease reports as self-reported general health ratings move from 1 (Excellent) to 5 (Poor).")
            else:
                st.info("Run training notebook to generate health rating comparison plot.")
                
            st.markdown("#### 4. Model Performance comparison (ROC Curve)")
            if os.path.exists("roc_curve_comparison.png"):
                st.image("roc_curve_comparison.png", caption="ROC Curve showing trade-off between True Positive Rate (Sensitivity) and False Positive Rate. Random Forest achieves ~80.2% AUC.")
            else:
                st.info("Run training notebook to generate ROC curve comparison plot.")

    with tab_info:
        st.subheader("About the CDC BRFSS Dataset")
        st.markdown("""
        The **Behavioral Risk Factor Surveillance System (BRFSS)** is the nation's premier system of health-related telephone surveys that collect state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services.
        
        This application uses a cleaned subset of the **2015 BRFSS survey** containing **253,680 records** with **21 health indicators** covering:
        1. **Biological and demographic markers:** Age, Sex, BMI.
        2. **Underlying conditions:** High Blood Pressure, High Cholesterol, Diabetes, history of Stroke.
        3. **Behavioral habits:** Smoking, Alcohol consumption, Fruit/Vegetable intake, Physical activity.
        4. **Socio-economic factors:** Education, Income, Healthcare insurance coverage, cost barrier to visit doctors.
        5. **Subjective health:** Self-rated general health, physical health days, mental health days.
        
        *Disclaimer: This tool is for educational purposes only. It should not be used as a replacement for professional medical advice, clinical diagnostics, or treatment.*
        """)
