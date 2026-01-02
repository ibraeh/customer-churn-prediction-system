import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    model = joblib.load('models/xgb_churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    return model, scaler, feature_names, explainer

model, scaler, feature_names, explainer = load_models()

# Title and description
st.title("🎯 Customer Churn Prediction System")
st.markdown("""
This app predicts whether a customer is likely to churn and identifies the key factors influencing the prediction.
Upload customer data or use the interactive form below.
""")

# Sidebar
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Model Insights"])

# ============================================
# PAGE 1: Single Prediction
# ============================================
if page == "Single Prediction":
    st.header("Single Customer Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    with col3:
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", 
                                      "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure))
    
    if st.button("🔮 Predict Churn", type="primary"):
        # Create input dataframe
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        df = pd.DataFrame([input_data])
        
        # Preprocess
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Align with training features
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_names]
        
        # Scale numerical features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
        
        # Predict
        prediction = model.predict(df_encoded)[0]
        probability = model.predict_proba(df_encoded)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            churn_status = "Will Churn ❌" if prediction == 1 else "Will Stay ✅"
            st.metric("Prediction", churn_status)
        
        with col2:
            st.metric("Churn Probability", f"{probability[1]:.1%}")
        
        with col3:
            risk_level = "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.4 else "Low"
            st.metric("Risk Level", risk_level)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if probability[1] > 0.5 else "darkgreen"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # SHAP explanation
        st.subheader("🔍 What's Driving This Prediction?")
        
        shap_values = explainer.shap_values(df_encoded)
        
        # Create SHAP force plot
        fig, ax = plt.subplots(figsize=(12, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            df_encoded.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig, bbox_inches='tight')
        
        # Top factors
        feature_importance = pd.DataFrame({
            'Feature': df_encoded.columns,
            'Impact': np.abs(shap_values[0])
        }).sort_values('Impact', ascending=False).head(5)
        
        st.dataframe(feature_importance, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Retention Recommendations")
        if prediction == 1:
            recommendations = []
            if contract == "Month-to-month":
                recommendations.append("✨ Offer a discount for switching to annual contract")
            if tenure < 12:
                recommendations.append("🎁 Provide loyalty rewards for new customers")
            if monthly_charges > 70:
                recommendations.append("💰 Review pricing or offer bundle discount")
            if internet_service == "Fiber optic" and online_security == "No":
                recommendations.append("🔒 Promote online security add-on services")
            
            for rec in recommendations:
                st.success(rec)
        else:
            st.info("✅ Customer appears stable. Continue providing excellent service!")

# ============================================
# PAGE 2: Batch Prediction
# ============================================
elif page == "Batch Prediction":
    st.header("Batch Customer Prediction")
    st.markdown("Upload a CSV file with customer data to get predictions for multiple customers.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_batch)} customers")
        st.dataframe(df_batch.head())
        
        if st.button("Run Batch Prediction"):
            # Preprocess
            df_processed = df_batch.copy()
            if 'customerID' in df_processed.columns:
                customer_ids = df_processed['customerID']
                df_processed = df_processed.drop('customerID', axis=1)
            
            # Encode
            categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
            if 'Churn' in categorical_cols:
                categorical_cols.remove('Churn')
            
            df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
            
            # Align features
            for col in feature_names:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[feature_names]
            
            # Scale
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
            
            # Predict
            predictions = model.predict(df_encoded)
            probabilities = model.predict_proba(df_encoded)[:, 1]
            
            # Results
            results_df = pd.DataFrame({
                'Customer ID': customer_ids if 'customerID' in df_batch.columns else range(len(df_batch)),
                'Churn Prediction': ['Yes' if p == 1 else 'No' for p in predictions],
                'Churn Probability': [f"{p:.2%}" for p in probabilities],
                'Risk Level': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in probabilities]
            })
            
            st.subheader("Prediction Results")
            st.dataframe(results_df)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", len(results_df))
            with col2:
                churn_count = (predictions == 1).sum()
                st.metric("Predicted Churners", churn_count)
            with col3:
                churn_rate = (predictions == 1).mean()
                st.metric("Churn Rate", f"{churn_rate:.1%}")
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

# ============================================
# PAGE 3: Model Insights
# ============================================
else:
    st.header("Model Performance & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [0.81, 0.68, 0.54, 0.60, 0.85]  # Replace with actual values
        })
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        st.subheader("Business Impact")
        st.markdown("""
        **Potential ROI:**
        - Average customer lifetime value: $2,400
        - Early intervention cost: $150
        - Predicted churners this month: 450
        - **Potential savings: $1.01M** (if 50% retention)
        """)
    
    st.subheader("Feature Importance")
    st.image('images/feature_importance.png')
    
    st.subheader("SHAP Summary")
    st.image('images/shap_summary.png')

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model: XGBoost ")
