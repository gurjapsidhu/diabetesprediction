import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
            background-color: #f4f4f9;
        }
        .header {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        }
        .sub-header {
            text-align: center;
            font-size: 18px;
            color: #555;
            margin-bottom: 30px;
        }
        .section-title {
            font-size: 24px;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .sidebar-title {
            font-size: 20px;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .button {
            background-color: #28a745;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            width: auto;
        }
        .button:hover {
            background-color: #218838;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #777;
            margin-top: 30px;
        }
        .link {
            color: white;
            text-decoration: none;
        }
        .result {
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 20px;
        }
        .result.success {
            color: green;
        }
        .result.error {
            color: red;
        }
        </style>
        <div class="header">Diabetes Prediction App</div>
        <div class="sub-header">AI-powered tool to assess diabetes risk using health parameters</div>
        """,
        unsafe_allow_html=True
    )

    # Display developer image and about section
    col1, col2 = st.columns([1, 3])
    with col1:
        img = Image.open("img.jpeg")  # Ensure the image path is correct
        st.image(img, width=150, caption="Developer: Gurjap Singh")
    with col2:
        st.markdown(
            """
            <div style="font-size: 16px; color: #333;">
            <p><b>About:</b> I am Gurjap Singh, a machine learning enthusiast with a passion for creating impactful AI applications. This app is designed to help users gain insights into their health using advanced machine learning techniques.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Sidebar for input features with plus/minus buttons
    st.sidebar.markdown("<div class='sidebar-title'>Input Health Parameters</div>", unsafe_allow_html=True)
    preg = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3, step=1)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=199, value=117, step=1)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=72, step=1)
    skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=99, value=23, step=1)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=30, step=1)
    bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.1, value=32.0, step=0.1)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.3725, step=0.001)
    age = st.sidebar.number_input('Age', min_value=21, max_value=81, value=29, step=1)

    # Main section for prediction results
    st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)
    input_data = np.array([preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]).reshape(1, -1)
    scaled_input_data = scaler.transform(input_data)
    prediction = model.predict(scaled_input_data)

    if st.button("Predict", key="predict_button"):
        with st.spinner("Analyzing..."):
            if prediction == 1:
                st.markdown("<div class='result error'><i class='fas fa-exclamation-triangle'></i> This person has diabetes.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result success'><i class='fas fa-check-circle'></i> This person does not have diabetes.</div>", unsafe_allow_html=True)

    # Footer with white link text
    st.markdown(
        """
        <div class="footer">
            Created by Gurjap Singh | Contact: gurjapsidhu5666@gmail.com | <a href="https://linkedin.com/in/gurjapsingh" class="link" target="_blank">Connect on LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    app()
