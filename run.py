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
        st.write("🔍 Attempting to load model...")
        model = joblib.load('churn_model.joblib')
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {type(e).__name__} – {e}")
        return None

model = load_model()

# --- Feature Importance Function ---
def plot_feature_importance(model):
    """
    Extracts and plots the feature importances from the trained pipeline.
    """
    try:
        # Extract the preprocessor and classifier from the pipeline
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']

        # Get the feature names from the preprocessor
        # The transformers_ attribute holds the original column names
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][2]
        
        # Get the one-hot encoded feature names
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
        
        # Combine all feature names in the correct order
        all_feature_names = list(num_features) + list(ohe_feature_names)

        # Get the coefficients from the logistic regression model
        coefficients = classifier.coef_[0]

        # Create a DataFrame for easy plotting
        feature_importance = pd.DataFrame({'Feature': all_feature_names, 'Importance': coefficients})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        # Separate features that increase vs decrease churn
        positive_importance = feature_importance[feature_importance['Importance'] > 0].head(10)
        negative_importance = feature_importance[feature_importance['Importance'] < 0].tail(10)
        
        # Combine the top positive and negative features for plotting
        top_features = pd.concat([positive_importance, negative_importance]).sort_values(by='Importance', ascending=False)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='coolwarm', ax=ax)
        ax.set_title('Top Factors Influencing Churn', fontsize=16)
        ax.set_xlabel('Coefficient (Impact on Churn)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        # Add a note on the plot
        fig.text(0.5, 0.01, 'Features with positive coefficients increase churn likelihood; negative coefficients decrease it.', ha='center', fontsize=10, style='italic')
        plt.tight_layout(rect=[0, 0.03, 1, 1]) # Adjust layout to make space for the text

        return fig

    except Exception as e:
        st.warning(f"Could not generate feature importance plot. Error: {e}")
        return None


# --- UI Layout and Styling ---
if model:
    st.title("🚀 Customer Churn Prediction Dashboard")
    st.markdown("Enter a customer's details using the form below to get a real-time churn prediction. The model analyzes the customer's profile and services to estimate their likelihood of leaving.")

    col1, col2 = st.columns(2)

    # --- Input Fields for Customer Data ---
    with col1:
        st.header("Customer Demographics")
        gender = st.selectbox("Gender", ('Male', 'Female'))
        senior_citizen_radio = st.radio("Senior Citizen", ('No', 'Yes'), horizontal=True)
        senior_citizen = 1 if senior_citizen_radio == 'Yes' else 0
        partner = st.radio("Has a Partner?", ('No', 'Yes'), horizontal=True)
        dependents = st.radio("Has Dependents?", ('No', 'Yes'), horizontal=True)

    with col2:
        st.header("Billing & Contract")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ('Month-to-month', 'One year', 'Two year'))
        paperless_billing = st.radio("Paperless Billing?", ('Yes', 'No'), horizontal=True)
        payment_method = st.selectbox("Payment Method", ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 55.0, 0.05)
        total_charges = st.slider("Total Charges ($)", 18.0, 9000.0, 500.0, 0.1)

    st.divider()

    st.header("Subscribed Services")
    col3, col4, col5 = st.columns(3)

    with col3:
        phone_service = st.selectbox("Phone Service", ('Yes', 'No'))
        multiple_lines = st.selectbox("Multiple Lines", ('No', 'Yes', 'No phone service'))
        internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))

    with col4:
        online_security = st.selectbox("Online Security", ('No', 'Yes', 'No internet service'))
        online_backup = st.selectbox("Online Backup", ('No', 'Yes', 'No internet service'))
        device_protection = st.selectbox("Device Protection", ('No', 'Yes', 'No internet service'))

    with col5:
        tech_support = st.selectbox("Tech Support", ('No', 'Yes', 'No internet service'))
        streaming_tv = st.selectbox("Streaming TV", ('No', 'Yes', 'No internet service'))
        streaming_movies = st.selectbox("Streaming Movies", ('No', 'Yes', 'No internet service'))

    # --- Prediction Logic ---
    if st.button("Predict Churn", type="primary", use_container_width=True):
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
            time.sleep(1)
            prediction_proba = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result", divider='rainbow')
        churn_probability_percent = prediction_proba * 100

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if churn_probability_percent > 50:
                st.metric(label="Churn Probability", value=f"{churn_probability_percent:.2f}%", delta="High Risk", delta_color="inverse")
            elif churn_probability_percent > 20:
                st.metric(label="Churn Probability", value=f"{churn_probability_percent:.2f}%", delta="Moderate Risk", delta_color="inverse")
            else:
                st.metric(label="Churn Probability", value=f"{churn_probability_percent:.2f}%", delta="Low Risk", delta_color="normal")

        with res_col2:
            if churn_probability_percent > 50:
                st.error("🚨 **Action Recommended:** This customer is at a high risk of churning. Consider proactive retention strategies like offering a discount, a service upgrade, or a personalized support call.")
            elif churn_probability_percent > 20:
                st.warning("⚠️ **Monitor:** This customer shows some signs of potential churn. It might be beneficial to monitor their usage and satisfaction levels.")
            else:
                st.success("✅ **Healthy Customer:** This customer appears to be loyal. Continue providing excellent service.")

    # --- Feature Importance Section ---
    st.markdown("---")
    with st.expander("🔍 Click here to see the model's feature importance"):
        st.markdown("""
        This chart shows which factors have the biggest impact on the model's churn prediction.
        - **<span style='color:red;'>Red bars</span>**: Features that increase the likelihood of churn.
        - **<span style='color:blue;'>Blue bars</span>**: Features that decrease the likelihood of churn (promote retention).
        """, unsafe_allow_html=True)
        
        fig = plot_feature_importance(model)
        if fig:
            st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown("Built using scikit-learn and Streamlit.")
