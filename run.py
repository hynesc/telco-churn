import matplotlib
# FIX 1: Set the backend BEFORE importing pyplot to prevent display errors on the server.
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """A function to load the model to avoid reloading it on every interaction."""
    try:
        model = joblib.load('churn_model.joblib')
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {type(e).__name__} – {e}")
        return None

model = load_model()

# --- Feature Importance Function ---
@st.cache_data # FIX 2: Cache the plot function to prevent memory overloads and reboot loops.
def plot_feature_importance(_model):
    """
    Extracts and plots the feature importances from the trained pipeline.
    The _model argument is used to make this function cacheable.
    """
    try:
        preprocessor = _model.named_steps['preprocessor']
        classifier = _model.named_steps['classifier']
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][2]
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
        all_feature_names = list(num_features) + list(ohe_feature_names)
        coefficients = classifier.coef_[0]
        feature_importance = pd.DataFrame({'Feature': all_feature_names, 'Importance': coefficients})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        top_features = pd.concat([feature_importance.head(10), feature_importance.tail(10)])

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='coolwarm', ax=ax)
        ax.set_title('Top Factors Influencing Churn', fontsize=16)
        ax.set_xlabel('Coefficient (Impact on Churn)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"Could not generate feature importance plot. Error: {e}")
        return None

# --- UI Layout and Styling ---
if model:
    st.title("🚀 Customer Churn Prediction Dashboard")
    st.markdown("Enter a customer's details to predict their churn likelihood.")

    # Main form for user input
    with st.form("customer_input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ('Male', 'Female'))
            senior_citizen = st.radio("Senior Citizen", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
            partner = st.selectbox("Has a Partner?", ('Yes', 'No'))
            dependents = st.selectbox("Has Dependents?", ('Yes', 'No'))
        with col2:
            st.subheader("Account Info")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
            paperless_billing = st.selectbox("Paperless Billing?", ('Yes', 'No'))
            payment_method = st.selectbox("Payment Method", ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 55.0)
            total_charges = st.slider("Total Charges ($)", 18.0, 9000.0, 500.0)
        with col3:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ('Yes', 'No'))
            multiple_lines = st.selectbox("Multiple Lines", ('No', 'Yes', 'No phone service'))
            internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
            online_security = st.selectbox("Online Security", ('No', 'Yes', 'No internet service'))
            online_backup = st.selectbox("Online Backup", ('No', 'Yes', 'No internet service'))
            device_protection = st.selectbox("Device Protection", ('No', 'Yes', 'No internet service'))
            tech_support = st.selectbox("Tech Support", ('No', 'Yes', 'No internet service'))
            streaming_tv = st.selectbox("Streaming TV", ('No', 'Yes', 'No internet service'))
            streaming_movies = st.selectbox("Streaming Movies", ('No', 'Yes', 'No internet service'))

        # Submit button for the form
        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame({
            'gender': [gender], 'SeniorCitizen': [senior_citizen], 'Partner': [partner],
            'Dependents': [dependents], 'tenure': [tenure], 'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines], 'InternetService': [internet_service],
            'OnlineSecurity': [online_security], 'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection], 'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv], 'StreamingMovies': [streaming_movies],
            'Contract': [contract], 'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method], 'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        with st.spinner('🧠 Analyzing customer profile...'):
            prediction_proba = model.predict_proba(input_data)[0][1]
        st.subheader("Prediction Result", divider='rainbow')
        churn_probability_percent = prediction_proba * 100
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if churn_probability_percent > 50:
                st.metric(label="Churn Probability", value=f"{churn_probability_percent:.2f}%", delta="High Risk", delta_color="inverse")
            elif churn_probability_percent > 20:
                st.metric(label="Churn Probability", value=f"{churn_probability_percent:.2f}%", delta="Moderate Risk", delta_color="inverse")
            else:
                st.metric(label="Churn Probability", value=f"{churn_probability_percent:.2f}%", delta="Low Risk", delta_color="normal")
        with res_col2:
            if churn_probability_percent > 50:
                st.error("🚨 **Action Recommended:** High risk of churning. Consider proactive retention strategies.")
            elif churn_probability_percent > 20:
                st.warning("⚠️ **Monitor:** Potential churn risk. Monitor usage and satisfaction.")
            else:
                st.success("✅ **Healthy Customer:** Appears loyal. Continue providing excellent service.")

    st.markdown("---")
    with st.expander("🔍 Click to see the model's feature importance"):
        st.markdown("This chart shows which factors have the biggest impact on the model's churn prediction.")
        fig = plot_feature_importance(model)
        if fig:
            st.pyplot(fig)
else:
    st.error("The application could not start because the model file failed to load.")