import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
st.set_page_config(page_title="HealthRisk AI Pro", layout="wide", page_icon="🏥")

# Custom CSS for a clean "Medical" look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 HealthRisk AI: Clinical Decision Support")
st.markdown("---")

# Navigation
page = st.sidebar.radio("Clinical Workflow", 
    ["1. Data & Batch Processing", "2. Model Intelligence", "3. System Evaluation", "4. Individual Patient View"])

# -------------------------------
# Page 1: Data & Batch Processing
# -------------------------------
if page == "1. Data & Batch Processing":
    st.header("📂 Data Management")
    
    tab1, tab2 = st.tabs(["Training Data Upload", "Batch Prediction"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload Training CSV", type="csv", key="train_upload")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state["data"] = df
            st.success("Training data synchronized.")
            st.dataframe(df.head(5), use_container_width=True)

    with tab2:
        st.subheader("Mass Risk Assessment")
        if "model" not in st.session_state:
            st.warning("Please train a model in Step 2 before using Batch Prediction.")
        else:
            batch_file = st.file_uploader("Upload New Patient List (CSV)", type="csv", key="batch_upload")
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                # Keep only numeric columns that match training features
                features = st.session_state["features"]
                X_batch = batch_df[features].fillna(batch_df[features].median())
                
                predictions = st.session_state["model"].predict(X_batch)
                probabilities = st.session_state["model"].predict_proba(X_batch)[:, 1] # Risk Prob
                
                batch_df['Predicted_Risk'] = predictions
                batch_df['Risk_Probability'] = (probabilities * 100).round(2).astype(str) + '%'
                
                st.write("### Batch Results")
                st.dataframe(batch_df, use_container_width=True)
                
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Annotated Patient List", data=csv, file_name="risk_assessments.csv", mime="text/csv")

# -------------------------------
# Page 2: Model Intelligence
# -------------------------------
elif page == "2. Model Intelligence":
    st.header("⚙️ Machine Learning Pipeline")
    if "data" in st.session_state:
        df = st.session_state["data"]
        if "Risk_Level" not in df.columns:
            st.error("Target 'Risk_Level' missing.")
        else:
            X = df.drop("Risk_Level", axis=1).select_dtypes(include=[np.number])
            y = df["Risk_Level"]
            
            if st.button("🚀 Train Clinical Ensemble"):
                with st.spinner("Optimizing Ensemble Model..."):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    
                    rf = RandomForestClassifier(n_estimators=100)
                    gb = GradientBoostingClassifier(n_estimators=100)
                    lr = LogisticRegression(max_iter=1000)
                    
                    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft')
                    ensemble.fit(X_train, y_train)
                    
                    st.session_state.update({"model": ensemble, "X_test": X_test, "y_test": y_test, "features": X.columns.tolist()})
                    st.success("Ensemble optimization complete.")
    else:
        st.info("Upload training data in Step 1.")

# -------------------------------
# Page 3: System Evaluation
# -------------------------------
elif page == "3. System Evaluation":
    st.header("📊 Performance Analytics")
    if "model" in st.session_state:
        # Code from previous version for Confusion Matrix and Feature Importance...
        # Using a Plotly chart for Feature Importance
        rf_feat = st.session_state["model"].named_estimators_['rf'].feature_importances_
        fig = px.bar(x=rf_feat, y=st.session_state["features"], orientation='h', title="Global Risk Drivers")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model not trained.")

# -------------------------------
# Page 4: Individual Patient View
# -------------------------------
elif page == "4. Individual Patient View":
    st.header("🩺 Patient-Specific Analysis")
    if "model" in st.session_state:
        model = st.session_state["model"]
        features = st.session_state["features"]
        df = st.session_state["data"]

        input_col, display_col = st.columns([1, 2], gap="large")

        with input_col:
            st.subheader("Inputs")
            user_input = []
            for col_name in features:
                val = st.number_input(f"{col_name}", value=float(df[col_name].median()))
                user_input.append(val)

        with display_col:
            st.subheader("Risk Visualization")
            prob = model.predict_proba([user_input])[0][-1] * 100
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                gauge = {'axis': {'range': [0, 100]},
                         'steps': [{'range': [0, 35], 'color': "#2ecc71"},
                                   {'range': [35, 70], 'color': "#f1c40f"},
                                   {'range': [70, 100], 'color': "#e74c3c"}],
                         'bar': {'color': "#2c3e50"}},
                title = {'text': "Risk Probability %"}))
            st.plotly_chart(fig, use_container_width=True)
            
            if prob > 70: st.error("⚠️ HIGH RISK: Immediate clinical review required.")
            elif prob > 35: st.warning("🟡 MODERATE RISK: Schedule follow-up.")
            else: st.success("🟢 LOW RISK: Routine monitoring.")
    else:
        st.warning("Model not trained.")
