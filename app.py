import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Streamlit Dashboard Layout
# -------------------------------
st.set_page_config(page_title="Healthcare Risk Dashboard", layout="wide")

st.title("📊 Healthcare Risk Prediction Dashboard")
st.write("Predict patient risk levels using ensemble machine learning models.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Upload Data", "Model Training", "Evaluation", "Predict Risk"])

# -------------------------------
# Upload Data
# -------------------------------
if page == "Upload Data":
    st.header("Step 1: Upload Healthcare Dataset")
    uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
        st.write("Preview of Data:", data.head())

        st.session_state["data"] = data
    else:
        st.info("Please upload a CSV file to proceed.")

# -------------------------------
# Model Training
# -------------------------------
elif page == "Model Training":
    st.header("Step 2: Train Ensemble Models")

    if "data" in st.session_state:
        data = st.session_state["data"]

        if "Risk_Level" not in data.columns:
            st.error("Dataset must contain a 'Risk_Level' column as target.")
        else:
            X = data.drop("Risk_Level", axis=1)
            y = data["Risk_Level"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            lr = LogisticRegression(max_iter=500)

            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft'
            )
            ensemble.fit(X_train, y_train)

            st.session_state["model"] = ensemble
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test

            st.success("Models trained successfully!")
    else:
        st.warning("Please upload data first.")

# -------------------------------
# Evaluation
# -------------------------------
elif page == "Evaluation":
    st.header("Step 3: Model Evaluation")

    if "model" in st.session_state:
        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.subheader("Classification Report")
        st.dataframe(report_df)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please train the model first.")

# -------------------------------
# Predict Risk
# -------------------------------
elif page == "Predict Risk":
    st.header("Step 4: Try Your Own Input")

    if "model" in st.session_state and "data" in st.session_state:
        model = st.session_state["model"]
        data = st.session_state["data"]
        X = data.drop("Risk_Level", axis=1)

        user_input = []
        for col in X.columns:
            val = st.number_input(f"{col}", value=float(X[col].mean()))
            user_input.append(val)

        if st.button("Predict Risk"):
            prediction = model.predict([user_input])[0]
            st.success(f"Predicted Risk Level: **{prediction}**")
    else:
        st.warning("Please upload data and train the model first.")
