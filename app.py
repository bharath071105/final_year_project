import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Dashboard Configuration ---
st.set_page_config(page_title="HealthRisk AI Pro", layout="wide", page_icon="🏥")

# --- 2. Secure Authentication ---
# Credentials are now pulled from st.secrets (external config) instead of code
def check_password():
    if "password_correct" not in st.session_state:
        st.title("🔐 Clinical Access Portal")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            # This checks the secrets.toml file or Streamlit Cloud Secrets
            if user in st.secrets["passwords"] and pwd == st.secrets["passwords"][user]:
                st.session_state["password_correct"] = True
                st.session_state["current_user"] = user
                st.rerun()
            else:
                st.error("Invalid credentials.")
        return False
    return True

# --- 3. Audit Logging ---
def log_action(action, details):
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []
    st.session_state["audit_log"].append({
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": st.session_state.get("current_user", "Authorized User"),
        "Action": action,
        "Details": details
    })

# --- 4. Main Application Flow ---
if check_password():
    st.sidebar.title("🏥 HealthRisk AI")
    st.sidebar.write(f"Logged in: **{st.session_state['current_user']}**")
    
    page = st.sidebar.radio("Clinical Workflow", 
        ["1. Data & Batch Processing", "2. Model Intelligence", "3. System Evaluation", "4. Patient Diagnosis", "5. Audit Trail"])

    if st.sidebar.button("Log Out"):
        del st.session_state["password_correct"]
        st.rerun()

    # --- Page 1: Data & Batch ---
    if page == "1. Data & Batch Processing":
        st.header("📂 Data Management")
        t1, t2 = st.tabs(["Training Data", "Batch Risk Assessment"])
        
        with t1:
            uploaded_file = st.file_uploader("Upload Training CSV", type="csv")
            if uploaded_file:
                st.session_state["data"] = pd.read_csv(uploaded_file)
                st.success("Training data loaded.")
                st.dataframe(st.session_state["data"].head())

        with t2:
            if "model" not in st.session_state:
                st.warning("Please train a model in Step 2 first.")
            else:
                batch_file = st.file_uploader("Upload New Patient List (CSV)", type="csv")
                if batch_file:
                    batch_df = pd.read_csv(batch_file)
                    features = st.session_state["features"]
                    X_batch = batch_df[features].fillna(batch_df[features].median())
                    
                    preds = st.session_state["model"].predict(X_batch)
                    batch_df['Predicted_Risk'] = preds
                    st.dataframe(batch_df)
                    
                    log_action("Batch Processing", f"Processed {len(batch_df)} records")
                    st.download_button("📥 Download Results", batch_df.to_csv(index=False), "risk_results.csv")

    # --- Page 2: Model Training ---
    elif page == "2. Model Intelligence":
        st.header("⚙️ Machine Learning Pipeline")
        
        if "data" in st.session_state:
            df = st.session_state["data"]
            if st.button("🚀 Train Clinical Ensemble"):
                with st.spinner("Optimizing Models..."):
                    X = df.drop("Risk_Level", axis=1).select_dtypes(include=[np.number])
                    y = df["Risk_Level"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    
                    model = VotingClassifier(estimators=[
                        ('rf', RandomForestClassifier()), 
                        ('gb', GradientBoostingClassifier()), 
                        ('lr', LogisticRegression(max_iter=1000))
                    ], voting='soft')
                    model.fit(X_train, y_train)
                    
                    st.session_state.update({"model": model, "X_test": X_test, "y_test": y_test, "features": X.columns.tolist()})
                    log_action("Model Training", "Trained new ensemble model")
                    st.success("Ensemble optimization complete.")
        else:
            st.info("Please upload training data in Step 1.")

    # --- Page 3: Evaluation ---
    elif page == "3. System Evaluation":
        st.header("📊 Performance Analytics")
        if "model" in st.session_state:
            y_pred = st.session_state["model"].predict(st.session_state["X_test"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Classification Report")
                report_df = pd.DataFrame(classification_report(st.session_state["y_test"], y_pred, output_dict=True)).transpose()
                st.dataframe(report_df)
            
            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state["y_test"], y_pred)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No model found. Train model in Step 2.")

    # --- Page 4: Diagnosis ---
    elif page == "4. Patient Diagnosis":
        st.header("🩺 Patient Analysis")
        if "model" in st.session_state:
            cols = st.columns([1, 2], gap="large")
            user_input = []
            with cols[0]:
                st.subheader("Input Parameters")
                for f in st.session_state["features"]:
                    user_input.append(st.number_input(f, value=float(st.session_state["data"][f].median())))
            
            with cols[1]:
                st.subheader("Diagnostic Result")
                prob = st.session_state["model"].predict_proba([user_input])[0][-1] * 100
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob,
                    gauge={'axis': {'range': [0, 100]},
                           'steps': [{'range': [0, 35], 'color': "#2ecc71"}, 
                                     {'range': [35, 70], 'color': "#f1c40f"}, 
                                     {'range': [70, 100], 'color': "#e74c3c"}],
                           'bar': {'color': "black"}},
                    title={'text': "Risk Probability %"}))
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Verify & Log Diagnosis"):
                    log_action("Individual Diagnosis", f"Risk Prob: {prob:.2f}%")
                    st.success("Diagnosis added to Audit Trail.")
        else:
            st.warning("Model not trained.")

    # --- Page 5: Audit ---
    elif page == "5. Audit Trail":
        st.header("📋 Clinical Audit Log")
        if "audit_log" in st.session_state:
            log_df = pd.DataFrame(st.session_state["audit_log"])
            st.dataframe(log_df, use_container_width=True)
            st.download_button("📥 Export Logs", log_df.to_csv(index=False), "audit_trail.csv")
        else:
            st.info("No activity recorded yet.")
