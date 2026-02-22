import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Credit Wise Loan Predictor", layout="wide")

@st.cache_resource
def prepare_model_and_ui():
    # 1. Load Data
    df = pd.read_csv("loan_approval_data.csv")
    if "Applicant_ID" in df.columns:
        df = df.drop("Applicant_ID", axis=1)

    target_col = "Loan_Approved"
    
    # 2. Identify Columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in cat_cols: 
        cat_cols.remove(target_col)
        
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col in num_cols: 
        num_cols.remove(target_col)

    # 3. Impute Missing Values (Required before finding mins/maxs)
    num_imp = SimpleImputer(strategy="mean")
    if num_cols: 
        df[num_cols] = num_imp.fit_transform(df[num_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    if cat_cols: 
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    # 4. Generate UI Configuration dynamically to avoid hidden 0s
    ui_config = {}
    for col in num_cols:
        ui_config[col] = {
            "type": "number",
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean())
        }
    for col in cat_cols:
        ui_config[col] = {
            "type": "category",
            "options": df[col].unique().tolist()
        }

    # 5. Target Encoding
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])

    # 6. Specific feature encoding from your notebook
    le_edu = None
    if "Education_Level" in df.columns:
        le_edu = LabelEncoder()
        df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])

    cols_to_ohe = ["Employment_Status", "Marital_Status", "Loan_Purpose", 
                   "Property_Area", "Gender", "Education_Level", "Employer_Category"]
    cols_to_ohe = [c for c in cols_to_ohe if c in df.columns]

    ohe = None
    if cols_to_ohe:
        ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        encoded = ohe.fit_transform(df[cols_to_ohe])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(), index=df.index)
        df = pd.concat([df.drop(columns=cols_to_ohe), encoded_df], axis=1)

    # 7. Feature Engineering
    if "DTI_Ratio" in df.columns:
        df["DTI_Ratio_sq"] = df["DTI_Ratio"]**2
    if "Credit_Score" in df.columns:
        df["Credit_Score_sq"] = df["Credit_Score"]**2

    # 8. Train/Test Prep
    x = df.drop(columns=[target_col, "Credit_Score", "DTI_Ratio"], errors='ignore')
    y = df[target_col]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    nb_model = GaussianNB()
    nb_model.fit(x_scaled, y)

    return nb_model, scaler, le_target, le_edu, ohe, cols_to_ohe, x.columns.tolist(), ui_config, num_cols, cat_cols

# --- Execute Training ---
try:
    model, scaler, le_target, le_edu, ohe, cols_to_ohe, feature_columns, ui_config, num_cols, cat_cols = prepare_model_and_ui()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Build Dynamic UI ---
st.title("🏦 Credit Wise Loan Approval System")
st.write("Enter the applicant's details below to predict if their loan will be approved.")

st.sidebar.header("Applicant Information")

user_input = {}

# Dynamically generate inputs for EVERY column in the dataset
for col in num_cols:
    user_input[col] = st.sidebar.number_input(
        f"{col.replace('_', ' ')}", 
        min_value=ui_config[col]["min"], 
        max_value=ui_config[col]["max"], 
        value=ui_config[col]["mean"]
    )

for col in cat_cols:
    user_input[col] = st.sidebar.selectbox(
        f"{col.replace('_', ' ')}", 
        ui_config[col]["options"]
    )

if st.sidebar.button("Predict Loan Status"):
    try:
        # 1. Convert user input to DataFrame
        input_df = pd.DataFrame([user_input])

        # 2. Apply Encoders
        if le_edu is not None and "Education_Level" in input_df.columns:
            input_df["Education_Level"] = le_edu.transform(input_df["Education_Level"])

        if ohe is not None and cols_to_ohe:
            encoded_input = ohe.transform(input_df[cols_to_ohe])
            encoded_df = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out())
            input_df = pd.concat([input_df.drop(columns=cols_to_ohe), encoded_df], axis=1)

        # 3. Apply Feature Engineering
        if "DTI_Ratio" in input_df.columns:
            input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"]**2
        if "Credit_Score" in input_df.columns:
            input_df["Credit_Score_sq"] = input_df["Credit_Score"]**2

        # Drop original features to match training setup
        input_df = input_df.drop(columns=["Credit_Score", "DTI_Ratio"], errors='ignore')

        # 4. Strict Column Alignment
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # 5. Scale & Predict
        input_scaled = scaler.transform(input_df)
        pred_label = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = max(probabilities) * 100
        
        pred_string = le_target.inverse_transform([pred_label])[0]
        positive_responses = ['yes', 'y', 'approved', 'approve', 'true', '1']
        
        st.subheader("Prediction Result")
        if str(pred_string).strip().lower() in positive_responses:
            st.success(f"🎉 Loan Approved! (Label: '{pred_string}')")
            st.write(f"**Model Confidence:** {confidence:.2f}%")
            st.balloons()
        else:
            st.error(f"❌ Loan Denied. (Label: '{pred_string}')")
            st.write(f"**Model Confidence:** {confidence:.2f}%")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
