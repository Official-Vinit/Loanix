import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Credit Wise Loan Predictor", layout="wide")

@st.cache_resource
def load_and_train():
    # 1. Load data
    df = pd.read_csv("loan_approval_data.csv")
    df = df.drop("Applicant_ID", axis=1, errors='ignore')

    # 2. Dynamically save original categorical options for the UI dropdowns
    cat_options = {}
    for col in df.select_dtypes(include=['object']).columns:
        cat_options[col] = df[col].dropna().unique().tolist()
        
    # 3. Impute missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    # 4. Encoding
    le = LabelEncoder()
    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])
    
    # Label encode Education_Level specifically, as done in your notebook
    le_edu = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])

    cols_to_ohe = ["Employment_Status", "Marital_Status", "Loan_Purpose", 
                   "Property_Area", "Gender", "Education_Level", "Employer_Category"]
    cols_to_ohe = [c for c in cols_to_ohe if c in df.columns]

    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    encoded = ohe.fit_transform(df[cols_to_ohe])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(), index=df.index)

    df = pd.concat([df.drop(columns=cols_to_ohe), encoded_df], axis=1)

    # 5. Feature Engineering
    df["DTI_Ratio_sq"] = df["DTI_Ratio"]**2
    df["Credit_Score_sq"] = df["Credit_Score"]**2

    # 6. Prepare features and target
    x = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]

    # 7. Scale data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # 8. Train Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(x_scaled, y)

    return nb_model, scaler, le_edu, ohe, cols_to_ohe, x.columns, cat_options

# Execute the training and extract the rules/options
try:
    model, scaler, le_edu, ohe, cols_to_ohe, feature_columns, cat_options = load_and_train()
except Exception as e:
    st.error(f"Error loading and training model: {e}")
    st.stop()

# --- UI BUILDING ---
st.title("🏦 Credit Wise Loan Approval System")
st.write("Enter the applicant's details below to predict if their loan will be approved.")

st.sidebar.header("Applicant Information")

# Helper function to safely get options from the dataset
def get_opts(col_name, default):
    return cat_options.get(col_name, default)

# UI Inputs (Dropdowns now perfectly match your CSV data)
app_income = st.sidebar.number_input("Applicant Income", min_value=0.0, value=50000.0)
credit_score = st.sidebar.number_input("Credit Score", min_value=300.0, max_value=850.0, value=700.0)
dti_ratio = st.sidebar.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3)
savings = st.sidebar.number_input("Savings", min_value=0.0, value=10000.0)

gender = st.sidebar.selectbox("Gender", get_opts("Gender", ["Male", "Female"]))
education = st.sidebar.selectbox("Education Level", get_opts("Education_Level", ["Graduate", "Not Graduate"]))
employment = st.sidebar.selectbox("Employment Status", get_opts("Employment_Status", ["Employed", "Unemployed"]))
marital = st.sidebar.selectbox("Marital Status", get_opts("Marital_Status", ["Yes", "No"]))
loan_purpose = st.sidebar.selectbox("Loan Purpose", get_opts("Loan_Purpose", ["Home", "Auto", "Personal"]))
property_area = st.sidebar.selectbox("Property Area", get_opts("Property_Area", ["Urban", "Rural", "Semiurban"]))
employer_cat = st.sidebar.selectbox("Employer Category", get_opts("Employer_Category", ["Private", "Government"]))

if st.sidebar.button("Predict Loan Status"):
    # 1. Create input dataframe
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

    try:
        # 2. Apply Label Encoding for Education Level exactly as in training
        input_data["Education_Level"] = le_edu.transform(input_data["Education_Level"])
        
        # 3. Apply One Hot Encoding
        encoded_input = ohe.transform(input_data[cols_to_ohe])
        encoded_input_df = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out())
        
        # Combine base data with OHE data
        input_data = pd.concat([input_data.drop(columns=cols_to_ohe), encoded_input_df], axis=1)

        # 4. Feature Engineering (Apply exact same math as training)
        input_data["DTI_Ratio_sq"] = input_data["DTI_Ratio"]**2
        input_data["Credit_Score_sq"] = input_data["Credit_Score"]**2

        # Drop original columns
        input_data = input_data.drop(columns=["Credit_Score", "DTI_Ratio"], errors='ignore')
        
        # 5. STRICT ALIGNMENT: Force the input dataframe to have the exact columns in the exact order as training
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

        # 6. Scale
        input_scaled = scaler.transform(input_data)

        # 7. Predict
        prediction = model.predict(input_scaled)
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("🎉 Loan Approved!")
            st.balloons()
        else:
            st.error("❌ Loan Denied.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction formatting: {e}")
