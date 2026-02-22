import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Credit Wise Loan Predictor", layout="wide")

# 1. Cache the training process so it only runs once when the app starts
@st.cache_resource
def train_model():
    # Load data
    df = pd.read_csv("loan_approval_data.csv")
    df = df.drop("Applicant_ID", axis=1, errors='ignore')

    # Impute missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    # Encode categorical variables
    le = LabelEncoder()
    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])
    df["Education_Level"] = le.fit_transform(df["Education_Level"])

    cols_to_ohe = ["Employment_Status", "Marital_Status", "Loan_Purpose", 
                   "Property_Area", "Gender", "Education_Level", "Employer_Category"]
    
    # Filter only columns that exist in the dataframe to avoid errors
    cols_to_ohe = [c for c in cols_to_ohe if c in df.columns]

    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    encoded = ohe.fit_transform(df[cols_to_ohe])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(), index=df.index)

    df = pd.concat([df.drop(columns=cols_to_ohe), encoded_df], axis=1)

    # Feature Engineering
    df["DTI_Ratio_sq"] = df["DTI_Ratio"]**2
    df["Credit_Score_sq"] = df["Credit_Score"]**2

    # Split data
    x = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]

    # Scale data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Train Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(x_scaled, y)

    # Return everything needed for prediction
    return nb_model, scaler, num_imp, cat_imp, le, ohe, cols_to_ohe, x.columns

# Load the trained model and preprocessors
model, scaler, num_imp, cat_imp, le, ohe, cols_to_ohe, feature_columns = train_model()

# 2. Build the Streamlit UI
st.title("🏦 Credit Wise Loan Approval System")
st.write("Enter the applicant's details below to predict if their loan will be approved.")

st.sidebar.header("Applicant Information")

# You can adjust these ranges based on your actual dataset
app_income = st.sidebar.number_input("Applicant Income", min_value=0, value=50000)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=700)
dti_ratio = st.sidebar.slider("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3)
savings = st.sidebar.number_input("Savings", min_value=0, value=10000)

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
employment = st.sidebar.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
marital = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Home", "Auto", "Personal", "Education"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
employer_cat = st.sidebar.selectbox("Employer Category", ["Private", "Government", "Self"])

if st.sidebar.button("Predict Loan Status"):
    # 3. Process the user input
    input_data = pd.DataFrame({
        "Applicant_Income": [app_income],
        "Credit_Score": [credit_score],
        "DTI_Ratio": [dti_ratio],
        "Savings": [savings],
        "Gender": [gender],
        "Education_Level": [education],
        "Employment_Status": [employment],
        "Marital_Status": [marital],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Employer_Category": [employer_cat]
    })

    # Apply LE
    # Note: We use a try-except to handle unseen labels gracefully if they differ from training
    try:
        input_data["Education_Level"] = le.transform(input_data["Education_Level"])
    except ValueError:
        input_data["Education_Level"] = 0 # Fallback 

    # Apply OHE
    encoded_input = ohe.transform(input_data[cols_to_ohe])
    encoded_input_df = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out())
    input_data = pd.concat([input_data.drop(columns=cols_to_ohe), encoded_input_df], axis=1)

    # Feature Engineering
    input_data["DTI_Ratio_sq"] = input_data["DTI_Ratio"]**2
    input_data["Credit_Score_sq"] = input_data["Credit_Score"]**2

    # Drop columns dropped in training and ensure column order matches
    input_data = input_data.drop(columns=["Credit_Score", "DTI_Ratio"], errors='ignore')
    
    # Add any missing OHE columns as 0 to match training data shape
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
            
    input_data = input_data[feature_columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("🎉 Loan Approved!")
        st.balloons()
    else:
        st.error("❌ Loan Denied.")