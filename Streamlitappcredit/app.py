import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "credit_model_final.pkl"
CONFIG_PATH = BASE_DIR / "model_config.pkl"
COST_CURVE_PATH = BASE_DIR / "cost_curve.png"

# ===== LOAD MODEL =====

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_config():
    return joblib.load(CONFIG_PATH)

model = load_model()
config = load_config()

# ===== STREAMLIT APP =====

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="💳", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Individual Analysis", "Batch Analysis", "Model Info"])

if page == "Home":
    st.title("💳 Credit Default Prediction System")
    st.write("**Business Impact:** $15.92M in annual savings (based on $10K avg loan)")
    
    st.header("Model Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{config['best_roc_auc']:.4f}")
    col2.metric("Optimal Threshold", f"{config['threshold']:.3f}")
    col3.metric("Recall", "93%")
    
    st.info("📊 Use **Individual Analysis** for single applicant assessment or **Batch Analysis** for bulk processing")
    
    st.markdown("---")
    st.subheader("Key Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("✅ **Cost-optimized** decision threshold (50:1 FN/FP ratio)")
        st.markdown("✅ **93% default detection** rate")
        st.markdown("✅ **Real-time risk scoring** for individual applicants")
    with col2:
        st.markdown("✅ **Batch processing** for high-volume screening")
        st.markdown("✅ **Regulatory-compliant** interpretability")
        st.markdown("✅ **$15.92M net annual benefit**")

elif page == "Individual Analysis":
    st.title("🔍 Individual Credit Risk Assessment")
    st.write("Enter applicant details to get instant risk prediction")
    
    # Example scenario buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Low Risk Example", use_container_width=True):
            st.session_state.example = "low"
    with col2:
        if st.button("📋 High Risk Example", use_container_width=True):
            st.session_state.example = "high"
    with col3:
        if st.button("📋 Stealth Defaulter Example", use_container_width=True):
            st.session_state.example = "stealth"
    
    # Initialize session state
    if 'example' not in st.session_state:
        st.session_state.example = None
    
    # Set default values based on example selection
    if st.session_state.example == "low":
        default_age = 45
        default_credit = 30.0
        default_debt = 0.3
        default_income = 6000
        default_dependents = 1
        default_30_59 = 0
        default_60_89 = 0
        default_90_plus = 0
        default_open_lines = 10
        default_real_estate = 1
        income_default_unknown = False
        dependents_default_unknown = False
    elif st.session_state.example == "high":
        default_age = 35
        default_credit = 85.0
        default_debt = 0.6
        default_income = 3500
        default_dependents = 2
        default_30_59 = 2
        default_60_89 = 1
        default_90_plus = 1
        default_open_lines = 5
        default_real_estate = 0
        income_default_unknown = False
        dependents_default_unknown = False
    elif st.session_state.example == "stealth":
        default_age = 60
        default_credit = 5.0
        default_debt = 0.25
        default_income = 5000
        default_dependents = 0
        default_30_59 = 0
        default_60_89 = 0
        default_90_plus = 0
        default_open_lines = 8
        default_real_estate = 2
        income_default_unknown = False
        dependents_default_unknown = False
    else:
        default_age = 45
        default_credit = 30.0
        default_debt = 0.3
        default_income = 5000
        default_dependents = 0
        default_30_59 = 0
        default_60_89 = 0
        default_90_plus = 0
        default_open_lines = 8
        default_real_estate = 1
        income_default_unknown = False
        dependents_default_unknown = False
    
    with st.form("credit_form"):
        st.subheader("📋 Applicant Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Personal & Financial**")
            age = st.number_input("Age", min_value=18, max_value=110, value=default_age, 
                                 help="Applicant's age in years")
            
            credit_used = st.slider("Credit Utilization (%)", min_value=0.0, max_value=100.0, 
                                   value=default_credit, step=0.1,
                                   help="Percentage of available credit currently used")
            
            debt_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=5.0, 
                                        value=default_debt, step=0.01,
                                        help="Monthly debt ÷ monthly income (typical: 0.2-0.5)")
            
            income_unknown = st.checkbox("Monthly Income Unknown", value=income_default_unknown)
            if not income_unknown:
                monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=1000000, 
                                               value=default_income, step=100)
            else:
                monthly_income = None
            
            dependents_unknown = st.checkbox("Number of Dependents Unknown", value=dependents_default_unknown)
            if not dependents_unknown:
                num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, 
                                                value=default_dependents, step=1)
            else:
                num_dependents = None
        
        with col2:
            st.markdown("**Late Payment History**")
            st.caption("Number of times past due in last 2 years")
            
            times_30_59 = st.number_input("30-59 Days Late", min_value=0, max_value=98, 
                                         value=default_30_59, step=1)
            
            times_60_89 = st.number_input("60-89 Days Late", min_value=0, max_value=20, 
                                         value=default_60_89, step=1)
            
            times_90_plus = st.number_input("90+ Days Late", min_value=0, max_value=17, 
                                           value=default_90_plus, step=1)
            
            st.markdown("**Credit Profile**")
            num_open_lines = st.number_input("Open Credit Lines & Loans", min_value=0, max_value=58, 
                                            value=default_open_lines, step=1,
                                            help="Total number of open credit accounts")
            
            num_real_estate = st.number_input("Real Estate Loans", min_value=0, max_value=54, 
                                             value=default_real_estate, step=1,
                                             help="Number of mortgage/real estate loans")
        
        submit = st.form_submit_button("🔍 Analyze Risk", use_container_width=True, type="primary")
    
    if submit:
        # Reset example selection after submission
        st.session_state.example = None
        
        # Create dataframe with proper column names
        input_data = pd.DataFrame({
            'Credit_used': [credit_used / 100],
            'age': [age],
            'Number_times_late_30_59_days': [times_30_59],
            'DebtRatio': [debt_ratio],
            'MonthlyIncome': [monthly_income],
            'NumberOfOpenCreditLinesAndLoans': [num_open_lines],
            'NumberOfTimes90DaysLate': [times_90_plus],
            'NumberRealEstateLoansOrLines': [num_real_estate],
            'NumberOfTime60-89DaysPastDueNotWorse': [times_60_89],
            'NumberOfDependents': [num_dependents]
        })
        
        # Get prediction
        with st.spinner("⚙️ Processing application..."):
            proba = model.predict_proba(input_data)[0, 1]
            threshold = config['threshold']
            decision = "REJECT" if proba >= threshold else "APPROVE"
        
        st.markdown("---")
        
        # === PREDICTION RESULT ===
        st.subheader("📊 Risk Assessment Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_pct = proba * 100
            if decision == "REJECT":
                st.error(f"### ❌ REJECT")
                st.metric("Default Risk", f"{risk_pct:.2f}%", 
                         delta=f"{risk_pct - threshold*100:.2f}pp above threshold", 
                         delta_color="inverse")
            else:
                st.success(f"### ✅ APPROVE")
                st.metric("Default Risk", f"{risk_pct:.2f}%", 
                         delta=f"{threshold*100 - risk_pct:.2f}pp below threshold", 
                         delta_color="normal")
        
        with col2:
            st.metric("Decision Threshold", f"{threshold*100:.2f}%")
        
        # === WARNING FOR STEALTH DEFAULTER ===
        if age >= 57 and credit_used < 10 and times_30_59 == 0 and times_60_89 == 0 and times_90_plus == 0:
            st.warning("⚠️ **Stealth Defaulter Pattern**: Age 57+, low utilization, clean history. Model has higher error rate on this segment.")
        
        # === RISK FACTOR BREAKDOWN ===
        st.markdown("---")
        st.subheader("🎯 Risk Factor Analysis")
        
        # Calculate derived features for display
        total_late = times_30_59 + times_60_89 + times_90_plus
        severity = times_30_59 + 2*times_60_89 + 3*times_90_plus
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Risk Indicators**")
            
            # Credit utilization assessment
            if credit_used > 80:
                st.error(f"🔴 Credit Utilization: {credit_used:.1f}% (HIGH RISK)")
            elif credit_used > 50:
                st.warning(f"🟡 Credit Utilization: {credit_used:.1f}% (MODERATE)")
            else:
                st.success(f"🟢 Credit Utilization: {credit_used:.1f}% (LOW RISK)")
            
            # Late payment assessment
            if total_late > 0:
                st.error(f"🔴 Total Late Events: {total_late} (Severity: {severity})")
            else:
                st.success(f"🟢 Clean Payment History")
            
            # Age assessment
            if age < 30:
                st.info(f"ℹ️ Age: {age} (Limited credit history typical)")
            elif age >= 57:
                st.info(f"ℹ️ Age: {age} (Monitor stealth defaulter pattern)")
            else:
                st.success(f"🟢 Age: {age} (Stable demographic)")
        
        with col2:
            st.markdown("**Financial Profile**")
            
            # Debt ratio
            if debt_ratio > 0.5:
                st.warning(f"🟡 Debt Ratio: {debt_ratio:.2f}")
            else:
                st.success(f"🟢 Debt Ratio: {debt_ratio:.2f}")
            
            # Income
            if monthly_income:
                if monthly_income < 3000:
                    st.warning(f"🟡 Monthly Income: ${monthly_income:,.0f}")
                else:
                    st.success(f"🟢 Monthly Income: ${monthly_income:,.0f}")
            else:
                st.info("ℹ️ Monthly Income: Not provided")
            
            # Credit profile
            st.info(f"📊 Open Credit Lines: {num_open_lines}")
            st.info(f"🏠 Real Estate Loans: {num_real_estate}")
        
        # === COMPARISON TO BENCHMARKS ===
        st.markdown("---")
        st.subheader("📈 Comparison to Typical Profiles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Typical APPROVED Profile**")
            st.write("• Credit Utilization: ~30%")
            st.write("• Age: 40-50 years")
            st.write("• Late Payments: 0")
            st.write("• Debt Ratio: 0.2-0.4")
            st.write("• Monthly Income: $5,000+")
        
        with col2:
            st.markdown("**Typical REJECTED Profile**")
            st.write("• Credit Utilization: 80%+")
            st.write("• Age: 30-40 years")
            st.write("• Late Payments: 2+")
            st.write("• Debt Ratio: 0.5+")
            st.write("• Monthly Income: $3,000-4,000")

elif page == "Batch Analysis":
    st.header("📦 Batch Prediction")
    st.write("Upload a CSV file with applicant data for bulk risk assessment")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(df)} records")
        
        with st.spinner("Processing applications..."):
            probas = model.predict_proba(df)[:, 1]
            
            threshold = config['threshold']
            df['Risk_Probability'] = probas
            df['Decision'] = np.where(probas >= threshold, 'REJECT', 'APPROVE')
        
        st.subheader("Predictions")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Applications", len(df))
        col2.metric("Approved", f"{(df['Decision']=='APPROVE').sum()}")
        col3.metric("Rejected", f"{(df['Decision']=='REJECT').sum()}")
        col4.metric("Avg Risk", f"{probas.mean()*100:.1f}%")
        
        # Risk Segmentation Table
        st.subheader("Risk Segmentation")
        
        # Calculate risk segments
        low_risk = (probas < 0.02).sum()
        medium_risk = ((probas >= 0.02) & (probas < 0.05)).sum()
        high_risk = ((probas >= 0.05) & (probas < 0.10)).sum()
        very_high_risk = (probas >= 0.10).sum()
        
        seg_data = pd.DataFrame({
            'Risk Category': ['Low Risk (0-2%)', 'Medium Risk (2-5%)', 'High Risk (5-10%)', 'Very High (>10%)'],
            'Count': [low_risk, medium_risk, high_risk, very_high_risk],
            'Percentage': [
                f"{low_risk/len(df)*100:.1f}%",
                f"{medium_risk/len(df)*100:.1f}%",
                f"{high_risk/len(df)*100:.1f}%",
                f"{very_high_risk/len(df)*100:.1f}%"
            ]
        })
        
        st.dataframe(seg_data, use_container_width=True, hide_index=True)
        
        # Risk distribution (collapsible)
        with st.expander("📊 View Risk Distribution Chart"):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(probas, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
            ax.set_xlabel('Default Probability')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Risk Scores')
            ax.legend()
            st.pyplot(fig)
            plt.close()
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="credit_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

elif page == "Model Info":
    st.header("📊 Model Performance Details")
    
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", "0.865")
    col2.metric("Recall", "93%")
    col3.metric("Precision", "12%")
    
    st.markdown("---")
    
    # Dual-tab financial analysis
    st.subheader("Financial Analysis")
    
    tab1, tab2 = st.tabs(["Model Training Costs", "Real-World Impact"])
    
    with tab1:
        st.write("**Cost parameters used during model optimization:**")
        st.write("- False Negative Cost: $5,000 (per missed default)")
        st.write("- False Positive Cost: $100 (per manual review)")
        st.write("- Cost Ratio: 50:1 (FN/FP)")
        st.write("")
        st.write("**Cost Breakdown:**")
        st.write("- False Negatives: 135 × $5,000 = $675,000")
        st.write("- False Positives: 13,950 × $100 = $1,395,000")
        st.write("- **Total Model Cost: $2,070,000**")
        st.write("")
        st.write("**Optimization Results:**")
        st.write("- Defaults Prevented: 1,867 × $5,000 = $9.34M")
        st.write("- Total Costs: $2.07M")
        st.write("- **Net Savings: $7.27M** (at training cost parameters)")
        st.caption("This validates the 50:1 cost optimization is working correctly (FN≈FP costs balanced)")
    
    with tab2:
        st.write("**Assuming $10,000 average loan size (₹8 Lakh in India):**")
        st.write("")
        st.write("**Benefits:**")
        st.write("- Defaults Caught: 1,867 applications")
        st.write("- Prevented Losses: 1,867 × $10,000 = **$18.67M**")
        st.write("")
        st.write("**Costs:**")
        st.write("- Review Cost: 13,950 × $100 = $1.40M")
        st.write("- Missed Defaults: 135 × $10,000 = $1.35M")
        st.write("- Total Costs: **$2.75M**")
        st.write("")
        st.write("**Net Benefit: $15.92M annually**")
        st.write("**ROI: 579% (5.8× return on operational costs)**")
        st.info("📍 This represents realistic ROI for lending scenarios with $10K average loan size")
    
    st.markdown("---")
    st.subheader("Hyperparameters")
    st.json(config['best_params'])
    
    st.subheader("Model Architecture")
    st.write("- **Algorithm:** LightGBM")
    st.write("- **Calibration:** Isotonic Regression (5-fold CV)")
    st.write("- **Threshold:** Cost-optimized (0.020)")
    st.write("- **Training Data:** 150K applications, 6.7% default rate")

    st.subheader("Cost Optimization Analysis")
    st.image(str(COST_CURVE_PATH), caption='Total Cost vs Threshold')
    st.write(f"**Optimal Threshold:** {config['threshold']:.3f}")
    
    st.markdown("---")
    st.subheader("Known Limitations")
    st.warning("⚠️ **Stealth Defaulter Blind Spot:** 7% false negative rate on older borrowers (age 57+) with low credit utilization and clean payment history.")
    
    st.info("ℹ️ Model trained on 2008-2011 data. Consider recalibration for current economic conditions.")