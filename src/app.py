# app.py — Streamlit demo for Churn model + ROI
import streamlit as st
import pandas as pd, joblib, os, numpy as np

DATA_PATH = "data/Telco-Customer-Churn.csv"
MODELS_DIR = "models"

st.set_page_config(page_title="Churn Prediction Demo", layout="wide")
st.title("Customer Churn Prediction — Interactive Demo")

# Sidebar controls
st.sidebar.header("Model & ROI settings")
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_pipeline.joblib")]
model_choice = st.sidebar.selectbox("Choose model pipeline", model_files)
threshold = st.sidebar.slider("Probability threshold", 0.01, 0.99, 0.5, 0.01)
cost_per_target = st.sidebar.number_input("Campaign cost per targeted customer ($)", value=50.0)
conversion = st.sidebar.number_input("Conversion rate (fraction)", value=0.20)
top_n = st.sidebar.number_input("Top N at-risk to show", min_value=5, max_value=500, value=20)

st.sidebar.markdown("**Usage**: run training first (`python src/train.py ...`) to create models/ and reports/")

# Load data & model
@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

data = load_data()
pipe = load_model(os.path.join(MODELS_DIR, model_choice))

# Prepare table with probabilities
X = data.drop(columns=['Churn','customerID'], errors='ignore')
data['churn_proba'] = pipe.predict_proba(X)[:,1]
data_sorted = data.sort_values('churn_proba', ascending=False).reset_index(drop=True)

st.subheader("Top at-risk customers")
st.dataframe(data_sorted.head(top_n)[['customerID','churn_proba','MonthlyCharges','tenure','Contract']])

# ROI calculation for chosen threshold
preds = (data['churn_proba'] >= threshold).astype(int)
y_true = data['Churn'].map({'Yes':1,'No':0})
tp = int(((preds==1) & (y_true==1)).sum())
fp = int(((preds==1) & (y_true==0)).sum())
total_targeted = tp + fp
expected_retained = conversion * tp
# baseline LTV (simple default — change to survival estimates if you have them)
avg_monthly_churners = data.loc[data['Churn']=='Yes', 'MonthlyCharges'].mean()
retained_avg_tenure = data.loc[data['Churn']=='No','tenure'].mean()
baseline_ltv = avg_monthly_churners * retained_avg_tenure

gross = expected_retained * baseline_ltv
cost = total_targeted * cost_per_target
net = gross - cost

col1, col2, col3 = st.columns(3)
col1.metric("Predicted targeted (positives)", f"{total_targeted}")
col2.metric("Estimated net savings ($)", f"${net:,.2f}")
col3.metric("Expected retained (approx.)", f"{expected_retained:.1f}")

st.markdown("### Export targeted list")
csv = data_sorted.loc[data_sorted['churn_proba']>=threshold, ['customerID','churn_proba','MonthlyCharges','tenure','Contract']].to_csv(index=False)
st.download_button("Download targeted CSV", csv, file_name="targeted_customers.csv", mime="text/csv")

st.markdown("---")
st.markdown("Generated using pre-trained pipeline. For production, host model securely and do not expose PII.")
