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
from openai import OpenAI

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="HealthRisk AI", page_icon="🏥", layout="wide")

# =====================================================
# CUSTOM UI STYLING
# =====================================================
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
</style>
""", unsafe_allow_html=True)

# =====================================================
# AUTHENTICATION
# =====================================================
def check_password():
    if "password_correct" not in st.session_state:
        st.title("🔐 Clinical Access Portal")

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

# =====================================================
# AUDIT LOGGING
# =====================================================
def log_action(action, details):
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []

    st.session_state["audit_log"].append({
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": st.session_state.get("current_user", "User"),
        "Action": action,
        "Details": details
    })

# =====================================================
# AI NAVIGATION HANDLER
# =====================================================
def handle_navigation_command(message):
    msg = message.lower()
    pages = ["Dashboard", "Risk Assessment", "Batch Processing",
             "Model Center", "System Evaluation", "Audit Log"]

    for p in pages:
        if p.lower() in msg:
            st.session_state["nav"] = p
            return f"Navigation detected. Redirecting to **{p}**."
    return None

# =====================================================
# MAIN APP
# =====================================================
if check_password():

    if "nav" not in st.session_state:
        st.session_state["nav"] = "Dashboard"

    page = st.selectbox(
        "",
        ["Dashboard", "Risk Assessment", "Batch Processing",
         "Model Center", "System Evaluation", "Audit Log"],
        index=["Dashboard", "Risk Assessment", "Batch Processing",
               "Model Center", "System Evaluation", "Audit Log"].index(st.session_state["nav"])
    )

    st.session_state["nav"] = page
    st.markdown("---")

    # =====================================================
    # DASHBOARD
    # =====================================================
    if page == "Dashboard":

        st.title("🏥 HealthRisk AI Platform")
        st.caption("Explainable Multi-Disease Risk Prediction System with Integrated AI Support")

        col1, col2, col3 = st.columns(3)
        col1.metric("System Status", "Operational")
        col2.metric("Active User", st.session_state["current_user"])
        col3.metric("Model Status", "Trained" if "model" in st.session_state else "Not Trained")

        st.markdown("### Model Transparency")
        st.info("""
        Ensemble Method: Soft Voting Classifier  
        Base Models:
        - Random Forest  
        - Gradient Boosting  
        - Logistic Regression  
        Training Split: 80/20  
        """)

        st.markdown("### Ethical Considerations")
        st.warning("""
        This system is developed for academic research purposes.
        It provides probabilistic risk estimation and does not constitute
        medical diagnosis or clinical decision-making authority.
        """)

    # =====================================================
    # MODEL CENTER
    # =====================================================
    elif page == "Model Center":

        st.header("⚙️ Model Training Center")

        uploaded_file = st.file_uploader("Upload Training Dataset (CSV)", type="csv")

        if uploaded_file:
            st.session_state["data"] = pd.read_csv(uploaded_file)
            st.success("Dataset Loaded")

        if "data" in st.session_state:

            if st.button("🚀 Train Ensemble Model"):

                df = st.session_state["data"]

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

                log_action("Model Training", "Ensemble trained")
                st.success("Model Training Completed")

    # =====================================================
    # RISK ASSESSMENT
    # =====================================================
    elif page == "Risk Assessment":

        st.header("🩺 Individual Risk Assessment")

        if "model" not in st.session_state:
            st.warning("Train model first.")
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
                st.session_state["last_risk_score"] = f"{risk_prob:.1f}%"

                st.subheader("Predicted Risk Score")
                st.metric("Risk Probability", f"{risk_prob:.1f}%")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_prob,
                    gauge={'axis': {'range': [0, 100]}}
                ))
                st.plotly_chart(fig, use_container_width=True)

                rf_model = model.named_estimators_['rf']
                contributions = rf_model.feature_importances_ * np.array(inputs)

                contrib_df = pd.DataFrame({
                    "Feature": features,
                    "Impact": contributions
                }).sort_values("Impact")

                st.subheader("Feature Contribution Analysis")
                fig_bar = px.bar(contrib_df, x="Impact", y="Feature", orientation="h")
                st.plotly_chart(fig_bar, use_container_width=True)

                log_action("Risk Assessment", f"Risk Score {risk_prob:.1f}%")

    # =====================================================
    # BATCH PROCESSING
    # =====================================================
    elif page == "Batch Processing":

        st.header("📂 Batch Risk Prediction")

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

        st.header("📊 Model Evaluation")

        if "model" not in st.session_state:
            st.warning("Train model first.")
        else:

            y_pred = st.session_state["model"].predict(st.session_state["X_test"])

            st.subheader("Classification Report")
            report_df = pd.DataFrame(
                classification_report(
                    st.session_state["y_test"],
                    y_pred,
                    output_dict=True
                )
            ).transpose()

            st.dataframe(report_df)

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(st.session_state["y_test"], y_pred)
            fig = px.imshow(cm, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # AUDIT LOG
    # =====================================================
    elif page == "Audit Log":

        st.header("📋 Audit Trail")

        if "audit_log" in st.session_state:
            log_df = pd.DataFrame(st.session_state["audit_log"])
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("No recorded activity yet.")

    # =====================================================
    # AI CLINICAL ASSISTANT (FYP LEVEL)
    # =====================================================
    st.markdown("---")
    st.markdown("## 🧠 AI Clinical Decision Support Assistant")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    client = OpenAI(api_key=st.secrets["openai"]["api_key"])

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Enter clinical or system-related query..."):

        st.session_state["chat_history"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        nav_response = handle_navigation_command(prompt)

        if nav_response:
            response = nav_response
        else:

            system_prompt = f"""
            You are a Clinical Decision Support Assistant.

            Latest risk score: {st.session_state.get("last_risk_score", "Not available")}

            Respond using:
            1. Risk Interpretation
            2. Contributing Factors
            3. Statistical Context
            4. Preventive Considerations
            5. Model Limitations

            Keep academic tone.
            Do not provide diagnosis.
            """

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *st.session_state["chat_history"]
                ],
                temperature=0.2
            )

            response = completion.choices[0].message.content
            response += "\n\n⚠️ This output is for academic research purposes only."

        log_action("AI Query", prompt)

        st.session_state["chat_history"].append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
