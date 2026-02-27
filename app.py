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

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="HealthRisk AI",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

.stButton>button {
    border-radius: 8px;
    height: 42px;
    font-weight: 600;
}

.metric-card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    text-align:center;
}

.risk-card {
    padding: 30px;
    border-radius: 15px;
    text-align:center;
    color: white;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------
def check_password():
    if "password_correct" not in st.session_state:
        st.markdown("## 🔐 Clinical Access Portal")
        col1, col2, col3 = st.columns([1,2,1])

        with col2:
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")

            if st.button("Login"):
                if user in st.secrets["passwords"] and pwd == st.secrets["passwords"][user]:
                    st.session_state["password_correct"] = True
                    st.session_state["current_user"] = user
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        return False
    return True

# ---------------------------------------------------
# AUDIT LOG
# ---------------------------------------------------
def log_action(action, details):
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []

    st.session_state["audit_log"].append({
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": st.session_state.get("current_user", "User"),
        "Action": action,
        "Details": details
    })

# ---------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------
if check_password():

    # ---------------- NAVIGATION ----------------
    page = st.selectbox(
        "",
        ["Dashboard", "Risk Assessment", "Batch Processing", "Model Center", "System Evaluation", "Audit Log"]
    )

    st.markdown("---")

    # =====================================================
    # DASHBOARD
    # =====================================================
    if page == "Dashboard":

        st.title("🏥 HealthRisk AI Platform")
        st.caption("AI-Powered Multi-Disease Risk Prediction System")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("System Status", "Operational", "✔ Secure")

        with col2:
            st.metric("Logged User", st.session_state["current_user"])

        with col3:
            st.metric("Models Loaded", "Ensemble Ready" if "model" in st.session_state else "Not Trained")

        st.markdown("### Platform Capabilities")
        st.info("""
        • Multi-Disease Risk Prediction  
        • Clinical Ensemble Modeling  
        • Batch Patient Processing  
        • Explainable Risk Drivers  
        • Secure Audit Logging  
        """)

    # =====================================================
    # MODEL CENTER
    # =====================================================
    elif page == "Model Center":

        st.header("⚙️ Model Intelligence Center")

        uploaded_file = st.file_uploader("Upload Training Dataset (CSV)", type="csv")

        if uploaded_file:
            st.session_state["data"] = pd.read_csv(uploaded_file)
            st.success("Dataset Loaded")

        if "data" in st.session_state:

            if st.button("🚀 Train Ensemble Model"):

                df = st.session_state["data"]

                with st.spinner("Training Clinical Ensemble..."):

                    X = df.drop("Risk_Level", axis=1).select_dtypes(include=[np.number])
                    y = df["Risk_Level"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    model = VotingClassifier(estimators=[
                        ('rf', RandomForestClassifier(n_estimators=100)),
                        ('gb', GradientBoostingClassifier()),
                        ('lr', LogisticRegression(max_iter=1000))
                    ], voting='soft')

                    model.fit(X_train, y_train)

                    st.session_state.update({
                        "model": model,
                        "X_test": X_test,
                        "y_test": y_test,
                        "features": X.columns.tolist()
                    })

                    log_action("Model Training", "Ensemble retrained")
                    st.success("Model Training Completed")

    # =====================================================
    # RISK ASSESSMENT
    # =====================================================
    elif page == "Risk Assessment":

        st.header("🩺 Patient Risk Assessment")

        if "model" not in st.session_state:
            st.warning("Please train the model first in Model Center.")
        else:

            model = st.session_state["model"]
            features = st.session_state["features"]
            df = st.session_state["data"]

            form = st.form("risk_form")

            with form:
                col1, col2 = st.columns(2)
                inputs = []

                for i, f in enumerate(features):
                    if i % 2 == 0:
                        with col1:
                            val = st.number_input(f, value=float(df[f].median()))
                    else:
                        with col2:
                            val = st.number_input(f, value=float(df[f].median()))
                    inputs.append(val)

                submit = st.form_submit_button("Run Risk Analysis")

            if submit:

                probs = model.predict_proba([inputs])[0]
                risk_prob = probs[-1] * 100

                if risk_prob >= 50:
                    color, label = "#e74c3c", "HIGH RISK"
                elif risk_prob >= 25:
                    color, label = "#f1c40f", "MODERATE RISK"
                else:
                    color, label = "#2ecc71", "LOW RISK"

                # Risk Card
                st.markdown(f"""
                <div class="risk-card" style="background:{color};">
                    <h1 style="font-size:48px;">{risk_prob:.1f}%</h1>
                    <p>{label}</p>
                </div>
                """, unsafe_allow_html=True)

                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_prob,
                    gauge={'axis': {'range': [0, 100]},
                           'steps': [
                               {'range': [0, 25], 'color': "#2ecc71"},
                               {'range': [25, 50], 'color': "#f1c40f"},
                               {'range': [50, 100], 'color': "#e74c3c"}
                           ]}
                ))

                st.plotly_chart(fig, use_container_width=True)

                # Risk Drivers
                st.subheader("Risk Driver Analysis")
                rf_model = model.named_estimators_['rf']
                contributions = rf_model.feature_importances_ * np.array(inputs)
                contrib_df = pd.DataFrame({
                    'Feature': features,
                    'Impact': contributions
                }).sort_values(by='Impact')

                fig_bar = px.bar(
                    contrib_df,
                    x='Impact',
                    y='Feature',
                    orientation='h',
                    color_discrete_sequence=[color]
                )

                st.plotly_chart(fig_bar, use_container_width=True)

                log_action("Individual Assessment", f"Risk Score: {risk_prob:.1f}%")

    # =====================================================
    # BATCH PROCESSING
    # =====================================================
    elif page == "Batch Processing":

        st.header("📂 Batch Patient Risk Analysis")

        if "model" not in st.session_state:
            st.warning("Train model first.")
        else:
            batch_file = st.file_uploader("Upload Patient CSV", type="csv")

            if batch_file:
                batch_df = pd.read_csv(batch_file)
                features = st.session_state["features"]

                X_batch = batch_df[features].fillna(batch_df[features].median())
                preds = st.session_state["model"].predict(X_batch)

                batch_df["Predicted_Risk"] = preds

                st.dataframe(batch_df, use_container_width=True)

                st.download_button(
                    "Download Results",
                    batch_df.to_csv(index=False),
                    "batch_results.csv"
                )

                log_action("Batch Processing", f"{len(batch_df)} records analyzed")

    # =====================================================
    # SYSTEM EVALUATION
    # =====================================================
    elif page == "System Evaluation":

        st.header("📊 Model Performance Analytics")

        if "model" not in st.session_state:
            st.warning("Train model first.")
        else:
            y_pred = st.session_state["model"].predict(st.session_state["X_test"])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Classification Report")
                report_df = pd.DataFrame(
                    classification_report(
                        st.session_state["y_test"],
                        y_pred,
                        output_dict=True
                    )
                ).transpose()

                st.dataframe(report_df)

            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state["y_test"], y_pred)
                fig = px.imshow(cm, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # AUDIT LOG
    # =====================================================
    elif page == "Audit Log":

        st.header("📋 Clinical Audit Trail")

        if "audit_log" in st.session_state:
            log_df = pd.DataFrame(st.session_state["audit_log"])
            st.dataframe(log_df, use_container_width=True)

            st.download_button(
                "Export Audit Log",
                log_df.to_csv(index=False),
                "audit_log.csv"
            )
        else:
            st.info("No recorded activity yet.")
