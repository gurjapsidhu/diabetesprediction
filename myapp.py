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
    # Custom Styling using st.markdown
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(to right, #ffffff, #f2f6fc);
        }
        .header {
            text-align: center;
            font-size: 3rem;
            color: #4CAF50;
            margin-top: 50px;
        }
        .container {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }
        .content {
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Header Title
    st.markdown('<div class="header"><h1>Diabetes Prediction App</h1></div>', unsafe_allow_html=True)

    # Information paragraph
    st.markdown("""
    This app predicts whether a person has diabetes based on various health metrics.
    Please input the following details to make a prediction.
    """)

    # Input fields in the sidebar for user to input data
    st.sidebar.header("Input Parameters")
    
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=100)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=800, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
    diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)

    # Button to make prediction
    if st.sidebar.button("Predict Diabetes"):
        # Prepare input data
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data_scaled)

        # Display results
        st.subheader("Prediction Result:")
        if prediction == 0:
            st.write("### No Diabetes")
        else:
            st.write("### Diabetes Detected")

# Run the app
if __name__ == "__main__":
    app()
