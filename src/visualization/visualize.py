import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)
