import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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

# Preprocess data
@st.cache_data
def preprocess_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test, scaler

# Train the model
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Load data and train model
diabetes_df = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(diabetes_df)
model = train_model(X_train, y_train)

# Main App
def app():
    # Header section
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #e0f7fa, #f2f6fc);
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .subheader {
            text-align: center;
            font-size: 1.5rem;
            color: #34495e;
        }
        .about {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            background: #f1f8ff;
            border-radius: 10px;
        }
        .button {
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 1rem;
            border-radius: 5px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #0056b3;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='header'>Diabetes Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>An AI-Powered Tool for Health Insights</div>", unsafe_allow_html=True)

    # Add image under the header
    img = Image.open("img.jpeg")
    st.image(img, use_column_width="auto", caption="AI-Powered Tool")

    # Sidebar for input
    st.sidebar.title("Input Features")
    preg = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 199, 117)
    bp = st.sidebar.slider("Blood Pressure", 0, 122, 72)
    skinthickness = st.sidebar.slider("Skin Thickness", 0, 99, 23)
    insulin = st.sidebar.slider("Insulin", 0, 846, 30)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider("Age", 21, 81, 29)

    # Prediction logic
    input_data = np.array([preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]).reshape(1, -1)
    prediction = model.predict(input_data)

    # Display prediction
    st.markdown("<h3 style='text-align: center;'>Prediction Result</h3>", unsafe_allow_html=True)
    if prediction == 1:
        st.warning("This person has diabetes.")
    else:
        st.success("This person does not have diabetes.")

    # About Section
    st.markdown(
        """
        <div class='about'>
            <h3>About the Developer</h3>
            <p>Gurjap Singh, 17 years old, AI and Machine Learning enthusiast.</p>
            <a href='https://linkedin.com/in/gurjapsingh' target='_blank' class='button'>Connect on LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )

app()
