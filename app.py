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

# --- 2. Authentication System ---
def check_password():
    """Returns True if the user has the correct password."""
    def password_entered():
        if st.session_state["username"] in st.secrets["passwords"] and \
           st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]:
            st.session_state["password_correct"] = True
            st.session_state["current_user"] = st.session_state["username"]
            del st.session_state["password"] 
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔐 Clinical Access Portal")
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔐 Clinical Access Portal")
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Credentials incorrect.")
        return False
    return True

# --- 3. Utility Functions ---
def log_action(action, details):
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []
    st.session_state["audit_log"].append({
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": st.session_state.get("current_user", "Unknown"),
        "Action": action,
        "Details": details
    })

# --- 4. Main Application ---
if check_password():
    st.sidebar.title("🏥 HealthRisk AI")
    st.sidebar.write(f"Logged in as: **{st.session_state['current_user']}**")
    
    page = st.sidebar.radio("Clinical Workflow", 
        ["1. Data & Batch Processing", "2. Model Intelligence", "3. System Evaluation", "4. Patient Diagnosis", "5. Audit Trail"])

    if st.sidebar.button("Log Out"):
        st.session_state["password_correct"] = False
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
                st.warning("Please train a model first.")
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
                with st.spinner("Training..."):
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
                    st.success("Model ready.")
        else:
            st.info("Upload data in Step 1.")

    # --- Page 3: Evaluation ---
    elif page == "3. System Evaluation":
        st.header("📊 Performance Analytics")
        if "model" in st.session_state:
            y_pred = st.session_state["model"].predict(st.session_state["X_test"])
            st.text("Classification Report")
            st.dataframe(pd.DataFrame(classification_report(st.session_state["y_test"], y_pred, output_dict=True)).transpose())
            
            # Confusion Matrix Visualization
            cm = confusion_matrix(st.session_state["y_test"], y_pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig)
        else:
            st.warning("Train model first.")

    # --- Page 4: Diagnosis ---
    elif page == "4. Patient Diagnosis":
        st.header("🩺 Patient Analysis")
        if "model" in st.session_state:
            cols = st.columns([1, 2])
            user_input = []
            with cols[0]:
                for f in st.session_state["features"]:
                    user_input.append(st.number_input(f, value=float(st.session_state["data"][f].median())))
            
            with cols[1]:
                prob = st.session_state["model"].predict_proba([user_input])[0][-1] * 100
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob,
                    title={'text': "Risk Probability %"},
                    gauge={'steps': [{'range': [0, 35], 'color': "green"}, {'range': [35, 70], 'color': "orange"}, {'range': [70, 100], 'color': "red"}]}
                ))
                st.plotly_chart(fig)
                if st.button("Verify & Log Diagnosis"):
                    log_action("Individual Diagnosis", f"Risk Prob: {prob:.2f}%")
                    st.success("Diagnosis logged to audit trail.")
        else:
            st.warning("Train model first.")

    # --- Page 5: Audit ---
    elif page == "5. Audit Trail":
        st.header("📋 Clinical Audit Log")
        if "audit_log" in st.session_state:
            st.table(pd.DataFrame(st.session_state["audit_log"]))
        else:
            st.info("No logs yet.")
