import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('TELCO.csv')
    return data

# Preprocess the dataset
def preprocess_data(df):
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df

def main():
    st.title("Telco Customer Churn Analysis")

    # Button to clear cache
    if st.button("Clear Cache"):
        clear_cache()
        st.success("Cache cleared!")

    # Load data
    data = load_data()

    # Customer Churn by Gender
    st.subheader("Customer Churn by Gender")
    gender_churn = data.groupby(['gender', 'Churn']).size().reset_index(name='counts')
    gender_fig = px.bar(gender_churn, x='gender', y='counts', color='Churn', barmode='group', title="Customer Churn by Gender")
    st.plotly_chart(gender_fig)

    # Customer Churn by Senior Citizen
    st.subheader("Customer Churn by Senior Citizen")
    senior_churn = data.groupby(['SeniorCitizen', 'Churn']).size().reset_index(name='counts')
    senior_churn['SeniorCitizen'] = senior_churn['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    senior_fig = px.bar(senior_churn, x='SeniorCitizen', y='counts', color='Churn', barmode='group', title="Customer Churn by Senior Citizen")
    st.plotly_chart(senior_fig)

    # Distribution of Monthly Charges
    st.subheader("Distribution of Monthly Charges")
    monthly_charges_fig = px.histogram(data, x='MonthlyCharges', nbins=50, title="Distribution of Monthly Charges")
    st.plotly_chart(monthly_charges_fig)

    # Customer Churn by Contract Type
    st.subheader("Customer Churn by Contract Type")
    contract_churn = data.groupby(['Contract', 'Churn']).size().reset_index(name='counts')
    contract_fig = px.bar(contract_churn, x='Contract', y='counts', color='Churn', barmode='group', title="Customer Churn by Contract Type")
    st.plotly_chart(contract_fig)

    # Tenure Distribution by Churn Status
    st.subheader("Tenure Distribution by Churn Status")
    tenure_fig = px.histogram(data, x='tenure', color='Churn', nbins=30, title="Tenure Distribution by Churn Status")
    st.plotly_chart(tenure_fig)

    # Churn Rate by Internet Service Type
    st.subheader("Churn Rate by Internet Service Type")
    internet_churn = data.groupby(['InternetService', 'Churn']).size().reset_index(name='counts')
    internet_fig = px.bar(internet_churn, x='InternetService', y='counts', color='Churn', barmode='group', title="Churn Rate by Internet Service Type")
    st.plotly_chart(internet_fig)

    # Churn Rate by Payment Method
    st.subheader("Churn Rate by Payment Method")
    payment_churn = data.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='counts')
    payment_fig = px.bar(payment_churn, x='PaymentMethod', y='counts', color='Churn', barmode='group', title="Churn Rate by Payment Method")
    st.plotly_chart(payment_fig)

if __name__ == '__main__':
        main()







# Main function
def main():
    st.title("Telco Customer Churn Prediction")

    # Load and preprocess data
    data = load_data()
    df = preprocess_data(data)
    
    # Split data into features and target
    X = df.drop(columns='Churn_Yes')
    y = df['Churn_Yes']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model selection
    st.sidebar.subheader("Choose Model")
    model_option = st.sidebar.selectbox("Model", ("Logistic Regression", "Support Vector Machine", "Random Forest", "XGBoost"))

    if model_option == "Logistic Regression":
        model = LogisticRegression()
    elif model_option == "Support Vector Machine":
        model = SVC(probability=True)
    elif model_option == "Random Forest":
        model = RandomForestClassifier()
    elif model_option == "XGBoost":
        model = XGBClassifier()

    # Train the selected model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    st.write("Classification Report:")
    st.write(class_report)

    # Make predictions for new data
    st.sidebar.subheader("Make a Prediction")
    def user_input_features():
        tenure = st.sidebar.slider('Tenure', min_value=0, max_value=72, value=12, step=1)
        MonthlyCharges = st.sidebar.number_input('Monthly Charges', min_value=0.0, max_value=120.0, value=30.0, step=0.1)
        TotalCharges = st.sidebar.number_input('Total Charges', min_value=0.0, max_value=9000.0, value=500.0, step=0.1)
        SeniorCitizen = st.sidebar.selectbox('Senior Citizen', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
        gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
        Partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
        Dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
        PhoneService = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
        MultipleLines = st.sidebar.selectbox('Multiple Lines', ('No', 'Yes', 'No phone service'))
        InternetService = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
        OnlineSecurity = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
        OnlineBackup = st.sidebar.selectbox('Online Backup', ('No', 'Yes', 'No internet service'))
        DeviceProtection = st.sidebar.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
        TechSupport = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
        StreamingTV = st.sidebar.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
        StreamingMovies = st.sidebar.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))
        Contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
        PaymentMethod = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

        data = {'tenure': tenure,
                'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges,
                'gender': gender,
                'Partner': Partner,
                'Dependents': Dependents,
                'PhoneService': PhoneService,
                'MultipleLines': MultipleLines,
                'InternetService': InternetService,
                'OnlineSecurity': OnlineSecurity,
                'OnlineBackup': OnlineBackup,
                'DeviceProtection': DeviceProtection,
                'TechSupport': TechSupport,
                'StreamingTV': StreamingTV,
                'StreamingMovies': StreamingMovies,
                'Contract': Contract,
                'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod}
        features = pd.DataFrame(data, index=[0])
        return features

    user_data = user_input_features()

    # Encode user input
    user_data = pd.get_dummies(user_data)
    user_data = user_data.reindex(columns=X.columns, fill_value=0)
    user_data = scaler.transform(user_data)

    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)

    st.sidebar.subheader("Prediction")
    st.sidebar.write('Probability of Churning: ', prediction_proba[0][1])
    st.sidebar.write('Prediction: ', 'Churn' if prediction[0] else 'Not Churn')

if __name__ == '__main__':
    main()