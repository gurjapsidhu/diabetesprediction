import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the function to load data
@st.cache_data
def load_data():
    return pd.read_csv('diabetes (2).csv')

# Cache the function to preprocess data
@st.cache_data
def preprocess_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test, scaler

# Cache the function to train the model
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Load or train model
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    diabetes_df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(diabetes_df)
    model = train_model(X_train, y_train)
    joblib.dump(model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

# Main App
def app():
    # Header section
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(to right, #f4f4f9, #e8f1f5);
        }
        .header {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .sub-header {
            text-align: center;
            font-size: 18px;
            color: #34495e;
            margin-bottom: 30px;
        }
        .section-title {
            font-size: 24px;
            color: #27ae60;
            margin-bottom: 10px;
        }
        .sidebar-title {
            font-size: 20px;
            color: #2980b9;
            margin-bottom: 10px;
        }
        .button {
            background-color: #FF5733;  /* A new vibrant color */
            color: white;
            padding: 20px 40px;  /* Increased padding for bigger button */
            border: none;
            border-radius: 10px;
            font-size: 20px;  /* Larger font size */
            cursor: pointer;
            transition: all 0.3s ease;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        .button:hover {
            background-color: #C70039;  /* Darker shade for hover effect */
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 30px;
        }
        .input-field {
            margin-bottom: 10px;
        }
        </style>
        <div class="header">Diabetes Prediction App</div>
        <div class="sub-header">AI-powered tool to assess diabetes risk using health parameters</div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for input features
    st.sidebar.markdown("<div class='sidebar-title'>Input Health Parameters</div>", unsafe_allow_html=True)
    preg = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3, step=1, format="%d", key="preg")
    glucose = st.sidebar.number_input('Glucose Level', min_value=0, max_value=200, value=117, step=1, key="glucose")
    bp = st.sidebar.number_input('Blood Pressure (mmHg)', min_value=0, max_value=130, value=72, step=1, key="bp")
    skinthickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=23, step=1, key="skin")
    insulin = st.sidebar.number_input('Insulin Level (µU/mL)', min_value=0, max_value=850, value=30, step=1, key="insulin")
    bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0, step=0.1, key="bmi")
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.3725, step=0.001, key="dpf")
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=29, step=1, key="age")

    # Main section for prediction results
    st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)
    input_data = np.array([preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]).reshape(1, -1)
    scaled_input_data = scaler.transform(input_data)
    prediction = model.predict(scaled_input_data)

    if st.button("Predict", key="predict_button", help="Click to predict if the person has diabetes"):
        with st.spinner("Analyzing..."):
            if prediction == 1:
                st.error("Prediction: This person has diabetes.", icon="🚨")
            else:
                st.success("Prediction: This person does not have diabetes.", icon="✅")

    # About section
    st.markdown(
        """
        <div class="footer">
            <p><b>About:</b> This app is developed to provide a quick and reliable way to assess diabetes risk. Leveraging machine learning, it aims to make healthcare insights accessible to everyone.</p>
            <p>Created by Gurjap Singh | Contact: gurjapsidhu5666@gmail.com</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    app()
