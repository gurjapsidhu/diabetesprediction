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
            background-color: #007BFF;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #0056b3;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #777;
            margin-top: 30px;
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
        img = Image.open("img.jpeg")
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

    # Sidebar for input features
    st.sidebar.markdown("<div class='sidebar-title'>Input Health Parameters</div>", unsafe_allow_html=True)
    preg = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # Main section for prediction results
    st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)
    input_data = np.array([preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]).reshape(1, -1)
    scaled_input_data = scaler.transform(input_data)
    prediction = model.predict(scaled_input_data)

    if st.button("Predict", key="predict_button"):
        with st.spinner("Analyzing..."):
            if prediction == 1:
                st.error("Prediction: This person has diabetes.", icon="🚨")
            else:
                st.success("Prediction: This person does not have diabetes.", icon="✅")

    # Footer
    st.markdown(
        """
        <div class="footer">
            Created by Gurjap Singh | Contact: gurjapsidhu5666@gmail.com
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    app()
