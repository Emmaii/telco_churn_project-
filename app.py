import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from src.features.preprocess import preprocess
from src.models.train import train_model

st.set_page_config(page_title="My Customer Prediction Model", layout="wide")
st.title("📊 My Customer Prediction Model")


if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None

menu = st.sidebar.radio("📌 Navigate", ["Upload Data", "Clean Data", "Analytics Dashboard", "Predict Customers"])

# -------------------
# Upload Data
# -------------------
if menu == "Upload Data":
    st.title("📤 Upload Your Customer Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

# -------------------
# Clean Data
# -------------------
elif menu == "Clean Data":
    st.title("🧹 Data Cleaning Tool")
    uploaded_file = st.file_uploader("Upload a CSV file to clean", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Original Data Preview")
        st.dataframe(df.head())

        cleaned_df = preprocess(df)

        st.subheader("✅ Cleaned Data Preview")
        st.dataframe(cleaned_df.head())

        st.markdown("""
        **What we did during cleaning:**
        - Removed unnecessary `customerID` column (if present).
        - Filled missing text values with `"Missing"`.
        - Converted text columns to numeric codes.
        - Filled missing numeric values with median.
        """)

        # Download button for cleaned data
        buffer = BytesIO()
        cleaned_df.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="💾 Download Cleaned Data",
            data=buffer,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

# -------------------
# Analytics Dashboard
# -------------------
elif menu == "Analytics Dashboard":
    st.title("📈 Analytics Dashboard")
    if st.session_state.df is None:
        st.warning("Please upload a CSV first.")
    else:
        df = st.session_state.df
        st.subheader("Churn Rate")
        churn_rate = df["Churn"].value_counts(normalize=True) * 100
        st.bar_chart(churn_rate)

        if "Contract" in df.columns:
            st.subheader("Contract Type Distribution")
            st.bar_chart(df["Contract"].value_counts())

        if "tenure" in df.columns:
            st.subheader("Tenure Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["tenure"], kde=True, ax=ax)
            st.pyplot(fig)

# -------------------
# Train Model
# -------------------
# -------------------
# Train Model
# -------------------
elif menu == "Train Model":
    st.title("🤖 Train Churn Prediction Model")
    if st.session_state.df is None:
        st.warning("Please upload a CSV first.")
    else:
        model, acc, prec, rec, cm, feature_importance = train_model(st.session_state.df)
        st.session_state.model = model
        st.session_state.feature_importance = feature_importance

        st.success("✅ Model trained successfully!")
        
        # Plain language communication
        st.subheader("Model Performance Summary")
        st.write(f"**Accuracy:** {acc:.2%} → My model got about {acc*100/100:.0f}% of predictions right.")
        st.write(f"**Precision:** {prec:.2%} → When predicting churn, it was correct {prec:.2%} of the time.")
        st.write(f"**Recall:** {rec:.2%} → It identified about {rec:.2%} of all actual churn cases.")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        st.pyplot(fig)

        st.subheader("Top Features Influencing Churn")
        st.bar_chart(feature_importance.set_index("Feature").head(10))
