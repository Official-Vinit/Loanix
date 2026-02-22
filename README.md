# 🏦 Credit Wise Loan System

## Overview
The **Credit Wise Loan System** is a machine learning web application designed to predict loan approval outcomes based on an applicant's financial background and demographic details. This project bridges data science and full-stack deployment, offering a seamless, interactive UI for real-time predictions. 

## 🚀 Live Demo
https://loanix.streamlit.app/

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit
* **Programming Language:** Python
* **Data Processing & EDA:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (Naive Bayes, K-Nearest Neighbors, Logistic Regression)

## 🧠 Machine Learning Pipeline
This project features a robust end-to-end data pipeline:
1.  **Data Preprocessing:** Handled missing values using `SimpleImputer` (mean for numerical, mode for categorical).
2.  **Encoding:** Applied `LabelEncoder` for ordinal data and `OneHotEncoder` for nominal categorical variables.
3.  **Feature Engineering:** Created non-linear features (`DTI_Ratio_sq`, `Credit_Score_sq`) to capture complex relationships in the financial metrics.
4.  **Scaling:** Standardized features using `StandardScaler` to optimize model performance.
5.  **Model Selection:** Evaluated Logistic Regression, KNN, and Naive Bayes. The **Gaussian Naive Bayes** model demonstrated the best performance metrics and was selected for the final deployment.

## 💻 Local Installation

To run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/credit-wise-loan-system.git](https://github.com/your-username/credit-wise-loan-system.git)
   cd credit-wise-loan-system
