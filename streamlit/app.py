import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== ONLY CUSTOM FUNCTIONS (needed for unpickling) =====

def fix_debt_ratio(df):
    df = df.copy()
    df['Monthly_income_missing'] = df['MonthlyIncome'].isnull().astype(int)
    df['Num_dependent_missing'] = df['NumberOfDependents'].isnull().astype(int)
    debt_median = df.loc[df['MonthlyIncome'].notnull(), 'DebtRatio'].median()
    df.loc[df['Monthly_income_missing'] == 1, 'DebtRatio'] = debt_median
    return df

def add_advanced_late_features(df):
    d30 = df["Number_times_late_30_59_days"]
    d60 = df["NumberOfTime60-89DaysPastDueNotWorse"]
    d90 = df["NumberOfTimes90DaysLate"]
    
    df["Total_Late_Events"] = d30 + d60 + d90
    df["Severity_Score"] = 1*d30 + 2*d60 + 3*d90
    df["Any_Severe_Late"] = (d60 + d90 > 0).astype(int)
    
    return df

def clip_outliers_iqr(df):
    df = df.copy()
    col_cap_iqr = ['Credit_used', 'age', 'DebtRatio',
                   'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                   'NumberRealEstateLoansOrLines', 'NumberOfDependents']
    
    for col in col_cap_iqr:
        q_1 = df[col].quantile(0.25)
        q_3 = df[col].quantile(0.75)
        iqr = q_3 - q_1
        lower_cap = q_1 - (1.5*iqr)
        upper_cap = q_3 + (1.5*iqr)
        df[col] = df[col].clip(lower_cap, upper_cap)
    return df

def clip_outliers_late(df):
    df = df.copy()
    col_to_cap = ["NumberOfTimes90DaysLate", "Number_times_late_30_59_days", 
                  "NumberOfTime60-89DaysPastDueNotWorse",
                  "Total_Late_Events", "Severity_Score", "Any_Severe_Late"]
    
    for col in col_to_cap:
        lower_cap = df[col].quantile(0)
        upper_cap = df[col].quantile(0.99)
        df[col] = df[col].clip(lower_cap, upper_cap)
    return df

# ===== LOAD MODEL =====

@st.cache_resource
def load_model():
    return joblib.load('credit_model_final.pkl')

@st.cache_resource
def load_config():
    return joblib.load('model_config.pkl')

model = load_model()
config = load_config()

# ===== STREAMLIT APP =====

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Batch Analysis", "Model Info"])

if page == "Home":
    st.title("Credit Default Prediction System")
    st.write("**Business Impact:** $16.5M in savings")
    
    st.header("Model Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{config['best_roc_auc']:.4f}")
    col2.metric("Optimal Threshold", f"{config['threshold']:.3f}")
    col3.metric("Recall", "93%")
    
    st.info("Upload applicant data in 'Batch Analysis' to get risk predictions")

elif page == "Batch Analysis":
    st.header("Batch Prediction")
    st.write("Upload a CSV file with applicant data for risk assessment")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(df)} records")
            
            # Predict
            probas = model.predict_proba(df)[:, 1]
            
            # Use saved threshold
            threshold = config['threshold']
            df['Risk_Probability'] = probas
            df['Decision'] = np.where(probas >= threshold, 'REJECT', 'APPROVE')
            
            # Show results
            st.subheader("Predictions")
            st.dataframe(df)
            
            # Summary stats
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Applications", len(df))
            col2.metric("Approved", f"{(df['Decision']=='APPROVE').sum()}")
            col3.metric("Rejected", f"{(df['Decision']=='REJECT').sum()}")
            
            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="credit_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
            import traceback
            st.code(traceback.format_exc())

elif page == "Model Info":
    st.header("Model Performance Details")
    
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", "0.865")
    col2.metric("Recall", "93%")
    col3.metric("Precision", "12%")
    
    st.subheader("Business Impact")
    st.write("- **Net Savings:** $16.5M")
    st.write("- **ROI:** $8 saved per $1 spent on reviews")
    st.write("- **Default Detection:** 93% of defaults caught")
    
    st.subheader("Hyperparameters")
    st.json(config['best_params'])
    
    st.subheader("Model Architecture")
    st.write("- Algorithm: LightGBM")
    st.write("- Calibration: Isotonic Regression (5-fold CV)")
    st.write("- Threshold: Cost-optimized (FP=$100, FN=$5000)")

    st.subheader("Cost Optimization Analysis")
    st.image('cost_curve.png', caption='Total Cost vs Threshold')
    st.write(f"**Optimal Threshold:** {config['threshold']:.3f}")
