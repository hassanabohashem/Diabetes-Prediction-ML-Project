import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# Note: StandardScaler and Pipeline are intentionally omitted to match the UN-SCALED KNN in your original script
import numpy as np
import warnings

# Hide warnings for a clean interface
warnings.filterwarnings('ignore', category=UserWarning)

# ===================================================================
# PAGE CONFIGURATION
# ===================================================================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# DATA LOADING AND MODEL TRAINING
# ===================================================================
@st.cache_resource
def train_models():
    """
    Loads data, preprocesses it, trains Decision Tree and KNN models.
    KNN is UN-SCALED to deliberately match the accuracy of the original script.
    """
    try:
        df = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        # Gracefully handle the missing file error
        return None
    
    # Imputation: Replace zeros with mean values (for columns specified in your original code)
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", 
                      "Insulin", "BMI", "DiabetesPedigreeFunction"]
    means = df[cols_with_zero].mean()
    imputation_means = means.to_dict()
    
    for col in cols_with_zero:
        df[col] = df[col].replace(0, means[col])
    
    # Outlier removal using 10th and 90th quantiles (as per your original code)
    low, high = 0.1, 0.9
    quant_df = df[["SkinThickness", "Insulin"]].quantile([low, high])
    df = df[
        (df["SkinThickness"] > quant_df.loc[low, "SkinThickness"]) &
        (df["SkinThickness"] < quant_df.loc[high, "SkinThickness"]) &
        (df["Insulin"] > quant_df.loc[low, "Insulin"]) &
        (df["Insulin"] < quant_df.loc[high, "Insulin"])
    ]
    df = df.dropna()
    
    # Train-test split (40% test size, random_state=30)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=30
    )
    
    # Train Decision Tree (max_depth=4)
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=0)
    dt_model.fit(X_train, y_train)
    dt_accuracy = dt_model.score(X_test, y_test)
    
    # Train KNN (K=21) WITHOUT Scaling to match the original script's accuracy (~0.71)
    knn_model = KNeighborsClassifier(n_neighbors=21)
    knn_model.fit(X_train, y_train)
    knn_accuracy = knn_model.score(X_test, y_test)
    
    return {
        'dt_model': dt_model,
        'knn_model': knn_model,
        'dt_accuracy': dt_accuracy,
        'knn_accuracy': knn_accuracy,
        'imputation_means': imputation_means,
        'feature_names': X.columns.tolist(),
        'dataset': df
    }

# Load models and data
model_data = train_models()

if model_data is None:
    st.error("⚠️ **Error**: 'diabetes.csv' file not found. Please ensure the dataset is in the correct location.")
    st.stop()

DT_MODEL = model_data['dt_model']
KNN_MODEL = model_data['knn_model']
DT_ACCURACY = model_data['dt_accuracy']
KNN_ACCURACY = model_data['knn_accuracy']
IMPUTATION_MEANS = model_data['imputation_means']
FEATURE_NAMES = model_data['feature_names']
DATASET = model_data['dataset']

# ===================================================================
# PREDICTION FUNCTION
# ===================================================================
def predict_diabetes(model, user_input_df):
    """Makes prediction based on user input with imputation applied."""
    user_input_processed = user_input_df.copy()
    
    for col, mean_val in IMPUTATION_MEANS.items():
        if col in user_input_processed.columns:
            # Impute 0 values in user input with the training set mean
            user_input_processed[col] = user_input_processed[col].replace(0, mean_val)
    
    # If the model is KNN (which should be unscaled here), we wrap it with a scaler
    # just for prediction if the user chooses the scaled option, to ensure 
    # the prediction works correctly, but we stick to the unscaled model for accuracy comparison.
    # Since we removed the Pipeline, we assume the selected model is the raw KNN model.
    
    prediction = model.predict(user_input_processed)
    probability = model.predict_proba(user_input_processed)
    
    return prediction[0], probability[0][1]

# ===================================================================
# SIDEBAR: MODEL INFORMATION & PERFORMANCE
# ===================================================================
with st.sidebar:
    st.title("📊 Model Analytics")
    st.markdown("---")
    
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        # DT Accuracy: ~0.76
        st.metric(
            label="Decision Tree",
            value=f"{DT_ACCURACY:.1%}",
            delta=f"{(DT_ACCURACY - 0.5):.1%}"
        )
    with col2:
        # KNN Accuracy: ~0.71 (UN-SCALED)
        st.metric(
            label="KNN (Unscaled)",
            value=f"{KNN_ACCURACY:.1%}",
            delta=f"{(KNN_ACCURACY - 0.5):.1%}"
        )
    
    st.markdown("---")
    st.subheader("📈 Dataset Statistics")
    total_samples = len(DATASET)
    diabetic_cases = DATASET['Outcome'].sum()
    non_diabetic_cases = total_samples - diabetic_cases
    
    st.info(f"""
**Total Samples**: {total_samples} 
**Diabetic Cases**: {diabetic_cases} ({DATASET['Outcome'].mean():.1%}) 
**Non-Diabetic Cases**: {non_diabetic_cases} ({(1-DATASET['Outcome'].mean()):.1%}) 
**Features**: {len(FEATURE_NAMES)}
    """)
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
This application uses machine learning to predict diabetes risk based on medical indicators.

**Models Used:**
- Decision Tree (Max Depth: 4)
- K-Nearest Neighbors (K=21, **Unscaled**)

**Preprocessing:**
- Zero value imputation
- Outlier removal (10th-90th percentile)
- 40% Test Split
    """)
    
    st.markdown("---")
    st.caption("⚕️ For informational purposes only. Consult a healthcare professional for medical advice.")

# ===================================================================
# MAIN APPLICATION
# ===================================================================
st.title("🏥 Diabetes Prediction System")
st.markdown("### Advanced Machine Learning-Based Risk Assessment")
st.markdown("---")

# Model selection
st.subheader("🤖 Select Prediction Model")
col1, col2 = st.columns([2, 1])

with col1:
    # Changed label from 'KNN (Scaled)' to 'KNN (Unscaled)'
    model_choice = st.radio(
        "Choose the machine learning algorithm:",
        options=['Decision Tree', 'KNN (Unscaled)'],
        horizontal=True,
        help="Decision Tree: Rule-based model | KNN: Distance-based model without feature scaling (matches original script)"
    )

with col2:
    if model_choice == 'Decision Tree':
        selected_model = DT_MODEL
        model_name = "Decision Tree"
        selected_accuracy = DT_ACCURACY
        st.success(f"✓ Accuracy: {DT_ACCURACY:.1%}")
    else:
        selected_model = KNN_MODEL
        model_name = "KNN (Unscaled)"
        selected_accuracy = KNN_ACCURACY
        st.success(f"✓ Accuracy: {KNN_ACCURACY:.1%}")

st.markdown("---")

# Patient data input
st.subheader("📋 Patient Information Input")
st.markdown("Enter patient medical data for diabetes risk prediction:")

# Create three columns for input fields
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 👤 Demographic & Pregnancy")
    Pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=17,
        value=3,
        step=1,
        help="Total number of pregnancies"
    )
    Age = st.number_input(
        "Age (years)",
        min_value=21,
        max_value=81,
        value=30,
        step=1,
        help="Patient's age in years"
    )
    
    st.markdown("##### 🩸 Blood Metrics")
    Glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0,
        max_value=200,
        value=120,
        step=1,
        help="Plasma glucose concentration (2 hours in oral glucose tolerance test)"
    )
    BloodPressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0,
        max_value=122,
        value=70,
        step=1,
        help="Diastolic blood pressure"
    )

with col2:
    st.markdown("##### 📏 Physical Measurements")
    SkinThickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=99,
        value=30,
        step=1,
        help="Triceps skin fold thickness"
    )
    BMI = st.number_input(
        "Body Mass Index (BMI)",
        min_value=0.0,
        max_value=67.1,
        value=32.0,
        step=0.1,
        format="%.1f",
        help="Body mass index (weight in kg/(height in m)²)"
    )
    
    st.markdown("##### 💉 Insulin Level")
    Insulin = st.number_input(
        "Insulin (μU/ml)",
        min_value=0,
        max_value=846,
        value=150,
        step=1,
        help="2-Hour serum insulin"
    )

with col3:
    st.markdown("##### 🧬 Genetic Factor")
    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.078,
        max_value=2.42,
        value=0.47,
        step=0.001,
        format="%.3f",
        help="Diabetes pedigree function (genetic influence score)"
    )
    
    st.markdown("##### 📊 Normal Ranges")
    st.info("""
**Reference Values:**
- Glucose: 70-100 mg/dL (fasting)
- Blood Pressure: 60-80 mm Hg
- BMI: 18.5-24.9 (normal)
- Insulin: 16-166 μU/ml
    """)

st.markdown("---")

# Prediction button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button(
        "🔬 Predict Diabetes Risk",
        use_container_width=True,
        type="primary"
    )

# ===================================================================
# PREDICTION RESULTS
# ===================================================================
if predict_button:
    # Collect user inputs into DataFrame
    user_data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })
    
    # Ensure column order matches training data
    user_data = user_data[FEATURE_NAMES]
    
    # Make prediction
    with st.spinner('Analyzing patient data...'):
        prediction, probability = predict_diabetes(selected_model, user_data.copy())
    
    st.markdown("---")
    st.markdown("## 🔬 Prediction Results")
    
    # Display result based on prediction
    if prediction == 1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                     padding: 30px; border-radius: 15px; border-left: 6px solid #f44336; 
                     box-shadow: 0 4px 12px rgba(244, 67, 54, 0.2);">
            <h2 style="color: #c62828; margin: 0;">⚠️ POSITIVE - High Risk</h2>
            <hr style="border: 1px solid #ef5350; margin: 15px 0;">
            <p style="font-size: 20px; font-weight: 600; color: #d32f2f;">
                Diabetes Probability: <span style="font-size: 24px;">{probability*100:.2f}%</span>
            </p>
            <p style="font-size: 16px; color: #b71c1c; margin-top: 15px;">
                The <strong>{model_name}</strong> model indicates a HIGH RISK of diabetes based on the provided medical data.
            </p>
            <p style="font-size: 14px; color: #c62828; margin-top: 10px; background-color: #ffebee; 
                      padding: 10px; border-radius: 5px;">
                ⚕️ <strong>Medical Consultation Required:</strong> This is an automated prediction. 
                Please consult a healthcare professional immediately for proper diagnosis and treatment.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                     padding: 30px; border-radius: 15px; border-left: 6px solid #4caf50; 
                     box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);">
            <h2 style="color: #2e7d32; margin: 0;">✅ NEGATIVE - Low Risk</h2>
            <hr style="border: 1px solid #66bb6a; margin: 15px 0;">
            <p style="font-size: 20px; font-weight: 600; color: #388e3c;">
                Non-Diabetes Probability: <span style="font-size: 24px;">{(1-probability)*100:.2f}%</span>
            </p>
            <p style="font-size: 16px; color: #1b5e20; margin-top: 15px;">
                The <strong>{model_name}</strong> model indicates a LOW RISK of diabetes based on the provided medical data.
            </p>
            <p style="font-size: 14px; color: #2e7d32; margin-top: 10px; background-color: #e8f5e9; 
                      padding: 10px; border-radius: 5px;">
                ℹ️ <strong>Note:</strong> Low risk does not guarantee absence of disease. 
                Regular health check-ups and maintaining a healthy lifestyle are recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display probability breakdown
    st.markdown("---")
    st.subheader("📊 Probability Breakdown")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Diabetic Probability",
            value=f"{probability*100:.2f}%",
            delta=f"{(probability - 0.5)*100:.2f}%" if probability > 0.5 else None
        )
    with col2:
        st.metric(
            label="Non-Diabetic Probability",
            value=f"{(1-probability)*100:.2f}%",
            delta=f"{((1-probability) - 0.5)*100:.2f}%" if (1-probability) > 0.5 else None
        )
    
    # Risk level indicator
    st.markdown("---")
    st.subheader("🎯 Risk Level Assessment")
    
    if probability < 0.3:
        risk_level = "LOW RISK"
        risk_color = "#4caf50"
        risk_icon = "✅"
    elif probability < 0.7:
        risk_level = "MODERATE RISK"
        risk_color = "#ff9800"
        risk_icon = "⚠️"
    else:
        risk_level = "HIGH RISK"
        risk_color = "#f44336"
        risk_icon = "🚨"
    
    st.markdown(f"""
    <div style="background-color: {risk_color}20; padding: 20px; border-radius: 10px; 
                 border: 2px solid {risk_color}; text-align: center;">
        <h3 style="color: {risk_color}; margin: 0;">{risk_icon} {risk_level}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display patient data used for prediction
    st.markdown("---")
    st.subheader("📄 Patient Data Summary")
    st.markdown("The following data was used for the prediction:")
    
    # Format the dataframe for better display
    display_df = user_data.copy()
    display_df.columns = ['Pregnancies', 'Glucose (mg/dL)', 'BP (mm Hg)', 
                          'Skin Thickness (mm)', 'Insulin (μU/ml)', 'BMI', 
                          'Pedigree Function', 'Age (years)']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Model information
    st.markdown("---")
    st.info(f"""
    **Model Used**: {model_name} 
    **Model Accuracy**: {selected_accuracy:.1%} 
    **Prediction Method**: {'Rule-based classification' if model_name == 'Decision Tree' else 'Distance-based classification without feature scaling'}
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Disclaimer:</strong> This tool is for educational and informational purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p style="font-size: 12px; margin-top: 10px;">
        © 2024 Diabetes Prediction System | Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)