import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

st.title("Healthcare Risk Prediction App")
st.write("Predict patient health risk levels using ensemble machine learning.")

# Upload dataset
uploaded_file = st.file_uploader("Upload healthcare dataset (CSV)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

    # Assume 'Risk_Level' is the target column
    if "Risk_Level" not in data.columns:
        st.error("Dataset must contain a 'Risk_Level' column as target.")
    else:
        X = data.drop("Risk_Level", axis=1)
        y = data["Risk_Level"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Models
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        lr = LogisticRegression(max_iter=500)

        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft'
        )
        ensemble.fit(X_train, y_train)

        st.success("Models trained successfully!")

        # Evaluation
        y_pred = ensemble.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("Classification Report:")
        st.dataframe(report_df)

        # User input
        st.subheader("Try Your Own Input")
        user_input = []
        for col in X.columns:
            val = st.number_input(f"{col}", value=float(X[col].mean()))
            user_input.append(val)

        if st.button("Predict Risk"):
            prediction = ensemble.predict([user_input])[0]
            st.write(f"Predicted Risk Level: **{prediction}**")
else:
    st.info("Please upload a CSV file to proceed.")
