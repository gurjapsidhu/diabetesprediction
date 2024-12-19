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
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Cache the function to load data
@st.cache
def load_data():
    diabetes_df = pd.read_csv('diabetes (2).csv')
    return diabetes_df

# Cache the function to preprocess data
@st.cache
def preprocess_data(df):
    # Split the data into input and target variables
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scale the input variables using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    return X_train, X_test, y_train, y_test, scaler

# Cache the function to train the model
@st.cache
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Load the pre-trained model if exists, else train the model
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    train_acc = None  # Accuracy not available without retraining
    test_acc = None
except:
    # Load data and preprocess
    diabetes_df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(diabetes_df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Save the trained model and scaler for future use
    joblib.dump(model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Calculate accuracy
    train_y_pred = model.predict(X_train)
    test_y_pred = model.predict(X_test)
    train_acc = accuracy_score(train_y_pred, y_train)
    test_acc = accuracy_score(test_y_pred, y_test)

# Streamlit app
def app():
    # Header section
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>Use this tool to predict diabetes based on health metrics.</p>", unsafe_allow_html=True)
    
    # Display an image in the header
    img = Image.open(r"img.jpeg")
    st.image(img, width=300, use_column_width=False)

    # Sidebar for input features
    st.sidebar.markdown("<h2 style='color: #4CAF50;'>Input Features</h2>", unsafe_allow_html=True)
    preg = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # Main section for results
    st.markdown("<h2 style='color: #4CAF50;'>Results</h2>", unsafe_allow_html=True)
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    input_data_nparray = np.asarray(input_data)
    reshaped_input_data = input_data_nparray.reshape(1, -1)
    scaled_input_data = scaler.transform(reshaped_input_data)  # Scale input data
    prediction = model.predict(scaled_input_data)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            if prediction == 1:
                st.error("This person has diabetes.")
            else:
                st.success("This person does not have diabetes.")

    # Add model accuracy section
    if train_acc and test_acc:
        st.markdown("<h2 style='color: #4CAF50;'>Model Accuracy</h2>", unsafe_allow_html=True)
        st.write(f"Train set accuracy: **{train_acc:.2f}**")
        st.write(f"Test set accuracy: **{test_acc:.2f}**")

    # Footer section
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>Developed by <b>Gurjap Singh</b>. Age: 17 years (2024). LinkedIn: <a href='https://www.linkedin.com/in/gurjap-singh-46696332a/' target='_blank'>Gurjap Singh</a></p>", unsafe_allow_html=True)

if __name__ == '__main__':
    app()
