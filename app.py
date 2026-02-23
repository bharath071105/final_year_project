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

# -------------------------------
# 1. Dashboard Configuration
# -------------------------------
st.set_page_config(page_title="HealthRisk AI Pro", layout="wide", page_icon="🏥")

# -------------------------------
# 2. Secure Authentication (via st.secrets)
# -------------------------------
def check_password():
    """Returns True if the user has the correct password from secrets.toml."""
    if "password_correct" not in st.session_state:
        st.title("🔐 Clinical Access Portal")
        user = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Login"):
            # Accessing credentials from the .streamlit/secrets.toml file
            if user in st.secrets["passwords"] and pwd == st.secrets["passwords"][user]:
                st.session_state["password_correct"] = True
                st.session_state["current_user"] = user
                st.rerun()
            else:
                st.error("Invalid credentials. Please contact your system administrator.")
        return False
    return True

# -------------------------------
# 3. Audit Logging Utility
# -------------------------------
def log_action(action, details):
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []
    st.session_state["audit_log"].append({
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": st.session_state.get("current_user", "Authorized User"),
        "Action": action,
        "Details": details
    })

# -------------------------------
# 4. Main Application
# -------------------------------
if check_password():
    # Sidebar Setup
    st.sidebar.title("🏥 HealthRisk AI")
    st.sidebar.write(f"Active Clinician: **{st.session_state['current_user']}**")
    
    page = st.sidebar.radio("Clinical Workflow", 
        ["1. Data & Batch Processing", "2. Model Intelligence", "3. System Evaluation", "4. Patient Diagnosis", "5. Audit Trail"])

    if st.sidebar.button("Log Out"):
        del st.session_state["password_correct"]
        st.rerun()

    # --- Page 1: Data & Batch Processing ---
    if page == "1. Data & Batch Processing":
        st.header("📂 Data Management")
        t1, t2 = st.tabs(["Training Data Upload", "Batch Risk Assessment"])
        
        with t1:
            uploaded_file = st.file_uploader("Upload Training CSV (Must include 'Risk_Level')", type="csv")
            if uploaded_file:
                st.session_state["data"] = pd.read_csv(uploaded_file)
                st.success("Training data synchronized successfully.")
                st.dataframe(st.session_state["data"].head(), use_container_width=True)

        with t2:
            if "model" not in st.session_state:
                st.warning("Action Required: Please train the ensemble model in Step 2 first.")
            else:
                batch_file = st.file_uploader("Upload Patient List for Batch Review", type="csv")
                if batch_file:
                    batch_df = pd.read_csv(batch_file)
                    features = st.session_state["features"]
                    # Automated preprocessing for batch
                    X_batch = batch_df[features].fillna(batch_df[features].median())
                    
                    preds = st.session_state["model"].predict(X_batch)
                    batch_df['Predicted_Risk'] = preds
                    st.write("### Batch Analysis Results")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    log_action("Batch Processing", f"Analyzed {len(batch_df)} patient records")
                    st.download_button("📥 Export Results as CSV", batch_df.to_csv(index=False), "batch_risk_results.csv")

    # --- Page 2: Model Intelligence ---
    elif page == "2. Model Intelligence":
        st.header("⚙️ Machine Learning Pipeline")
        st.info("System utilizes a Soft-Voting Ensemble: Random Forest + Gradient Boosting + Logistic Regression.")
        
        if "data" in st.session_state:
            df = st.session_state["data"]
            if st.button("🚀 Train Clinical Ensemble"):
                with st.spinner("Optimizing weights and evaluating sensitivity..."):
                    # Preprocessing
                    X = df.drop("Risk_Level", axis=1).select_dtypes(include=[np.number])
                    y = df["Risk_Level"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Ensemble Logic
                    model = VotingClassifier(estimators=[
                        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)), 
                        ('gb', GradientBoostingClassifier(random_state=42)), 
                        ('lr', LogisticRegression(max_iter=1000))
                    ], voting='soft')
                    model.fit(X_train, y_train)
                    
                    st.session_state.update({
                        "model": model, 
                        "X_test": X_test, 
                        "y_test": y_test, 
                        "features": X.columns.tolist()
                    })
                    log_action("Model Training", "Ensemble model updated with new data.")
                    st.success("Ensemble Intelligence Active.")
        else:
            st.info("Please upload training data in Step 1 to initialize the model.")

    # --- Page 3: System Evaluation ---
    elif page == "3. System Evaluation":
        st.header("📊 Performance Analytics")
        if "model" in st.session_state:
            y_pred = st.session_state["model"].predict(st.session_state["X_test"])
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Accuracy")
                report_df = pd.DataFrame(classification_report(st.session_state["y_test"], y_pred, output_dict=True)).transpose()
                st.dataframe(report_df)
                
            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state["y_test"], y_pred)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", 
                               labels=dict(x="Predicted Risk", y="Actual Risk"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model must be trained before performance metrics can be generated.")

    # --- Page 4: Patient Diagnosis ---
    elif page == "4. Patient Diagnosis":
        st.header("🩺 High-Sensitivity Diagnostic Dashboard")
        if "model" in st.session_state:
            model = st.session_state["model"]
            features = st.session_state["features"]
            df = st.session_state["data"]

            input_col, display_col = st.columns([1, 2], gap="large")

            with input_col:
                st.subheader("📋 Metrics Input")
                user_input = []
                with st.expander("Adjust Patient Parameters", expanded=True):
                    for f in features:
                        val = st.number_input(f, value=float(df[f].median()))
                        user_input.append(val)
            
            with display_col:
                probs = model.predict_proba([user_input])[0]
                risk_prob = probs[-1] * 100 
                
                # --- SENSITIVITY LOGIC: Alert triggers at 50% ---
                if risk_prob >= 50:
                    status_color, status_label = "#e74c3c", "CRITICAL / HIGH RISK"
                elif risk_prob >= 25:
                    status_color, status_label = "#f1c40f", "MODERATE / MONITORING"
                else:
                    status_color, status_label = "#2ecc71", "STABLE / LOW RISK"

                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    # Risk-o-meter Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=risk_prob,
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "black"},
                               'steps': [
                                   {'range': [0, 25], 'color': '#ebfaeb'},
                                   {'range': [25, 50], 'color': '#fff9e6'},
                                   {'range': [50, 100], 'color': '#ffe6e6'}],
                               'threshold': {'line': {'color': "red", 'width': 4}, 'value': 50}},
                        title={'text': "Risk Probability %"}))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with v_col2:
                    # Local Driver Analysis (Explainability)
                    rf_model = model.named_estimators_['rf']
                    contributions = rf_model.feature_importances_ * np.array(user_input)
                    contrib_df = pd.DataFrame({'Feature': features, 'Weight': contributions}).sort_values(by='Weight')
                    
                    fig_contrib = px.bar(contrib_df, x='Weight', y='Feature', orientation='h', 
                                         title="Primary Risk Drivers", color_discrete_sequence=[status_color])
                    st.plotly_chart(fig_contrib, use_container_width=True)

                # High-Visual Alert Card
                st.markdown(f"""
                    <div style="background-color:{status_color}; padding:25px; border-radius:10px; text-align:center; border: 2px solid rgba(0,0,0,0.1);">
                        <h2 style="color:white; margin:0; font-family:sans-serif;">{status_label}</h2>
                        <p style="color:white; margin:5px; font-size:1.2rem; opacity:0.9;">Confidence: {risk_prob:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("") # Spacer
                if st.button("💾 Finalize & Log Assessment"):
                    log_action("Individual Diagnosis", f"Risk: {risk_prob:.1f}% | Lead Factor: {contrib_df.iloc[-1]['Feature']}")
                    st.success("Clinical record saved to Audit Trail.")
        else:
            st.warning("Training required to access diagnostics.")

    # --- Page 5: Audit Trail ---
    elif page == "5. Audit Trail":
        st.header("📋 Clinical Audit Log")
        if "audit_log" in st.session_state:
            log_df = pd.DataFrame(st.session_state["audit_log"])
            st.dataframe(log_df, use_container_width=True)
            st.download_button("📥 Export System Logs", log_df.to_csv(index=False), "clinical_audit_trail.csv")
        else:
            st.info("No system activity recorded in this session.")
