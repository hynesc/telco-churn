# Customer Churn Prediction with Streamlit

This project demonstrates a complete machine learning workflow for predicting customer churn. It involves data cleaning, model training, feature importance analysis, and deployment of an interactive web application using Streamlit.

## Live Application

You can view the deployed interactive application here:
**[https://telco-churn-analysis.streamlit.app/](https://telco-churn-analysis.streamlit.app/)**

## Key Features

-   **Interactive Prediction**: Use sliders and dropdowns to input customer data and receive a churn prediction.
-   **Clear Results**: The app displays a clear, understandable prediction of whether a customer is likely to churn.
-   **Prediction Confidence**: Shows the model's confidence score (probability) for each prediction.
-   **Optimized Model**: Powered by a pruned logistic regression model that has been optimized through feature selection for efficiency and interpretability.

## 1. Problem Statement

The goal of this project is to build a model that can predict which customers of a telecom company are most likely to "churn" (cancel their subscription). By identifying these at-risk customers, the business can proactively offer incentives or support to retain them, which is often more cost-effective than acquiring new customers.

## 2. Dataset

This project uses the "Telco Customer Churn" dataset, a popular dataset for classification tasks. It contains 7043 customer records with 21 attributes, including:

* **Demographic Info**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Account Info**: `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* **Subscribed Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, etc.
* **Target Variable**: `Churn` (Yes/No)

## 3. Methodology

The project follows a standard data science workflow:

### a. Data Cleaning and Preprocessing

* The `TotalCharges` column was converted to a numeric type, and rows with missing values were dropped.
* The `customerID` column was removed as it is not a predictive feature.
* The target variable `Churn` was converted from "Yes"/"No" to a binary format (1/0).

### b. Model Training and Feature Selection

* A Logistic Regression model was chosen for its effectiveness and high interpretability.
* A `ColumnTransformer` pipeline was used to automatically apply `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.
* **Feature Importance Analysis**: The model's coefficients were analyzed to determine the impact of each feature on churn. It was discovered that `gender`, `PhoneService`, and `Partner` had a negligible impact on the model's predictive power.
* **Model Pruning**: A second model was trained with these unimportant features removed. A comparison of performance reports showed no significant drop in accuracy or F1-score.
* **Final Model**: The pruned model was selected for the final application as it provides the same performance with a simpler and more efficient structure. The final model achieved an **F1-score of 0.61** for predicting churners.

## 4. File Structure

The repository contains the following key files:

* `churn_model_training.ipynb`: A Jupyter Notebook containing all steps for data analysis, feature selection, model training, and evaluation.
* `run.py`: The main Streamlit application script that loads the trained model and serves the interactive UI.
* `churn_model.joblib`: The pre-trained and pruned machine learning model pipeline.
* `requirements.txt`: A list of all necessary Python libraries with pinned versions to ensure a consistent environment.
* `runtime.txt`: A configuration file that explicitly tells Streamlit Cloud to use Python 3.11, matching the training environment.
* `README.md`: This file.

## 5. How to Run Locally

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone github.com/hynesc/telco-churn
    cd telco-churn
    ```

2.  **Set Up a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the Model (Optional)**: A pre-trained `churn_model.joblib` is already included. However, if you wish to review the analysis or retrain the model yourself, you can run the `churn_model_training.ipynb` notebook in a Jupyter environment. The notebook will generate a new `churn_model.joblib` file.

5.  **Launch the Streamlit App**:
    ```bash
    streamlit run run.py
    ```
    Your web browser will automatically open with the interactive churn prediction dashboard.

## 6. Deployment

This application is designed for deployment on Streamlit Cloud. The `requirements.txt` and `runtime.txt` files are crucial for a successful deployment, as they ensure the cloud environment perfectly matches the training environment, preventing version conflicts.

## 7. Tools and Libraries

* **Data Analysis & Modeling**: Pandas, Scikit-learn, Jupyter
* **Web Application**: Streamlit
* **Plotting**: Matplotlib, Seaborn
