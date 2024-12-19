import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

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
    
    return X_train, X_test, y_train, y_test

# Cache the function to train the model
@st.cache
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Load the pre-trained model if exists, else train the model
try:
    model = joblib.load('diabetes_model.pkl')
except:
    # Load data and preprocess
    diabetes_df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(diabetes_df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Save the trained model for future use
    joblib.dump(model, 'diabetes_model.pkl')

# Make predictions on the training and testing sets
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

# Calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)

# Streamlit app
def app():
    st.title("Predict Diabetes")

    # Load and resize image
    img = Image.open(r"img.jpeg")
    img = img.resize((200, 200))  # Resize the image for faster loading
    st.image(img, width=200)

    st.title('Diabetes Prediction')

    # Create the input form for the user to input new data
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # Make a prediction based on the user input
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    input_data_nparray = np.asarray(input_data)
    reshaped_input_data = input_data_nparray.reshape(1, -1)
    prediction = model.predict(reshaped_input_data)

    # Display the prediction to the user
    st.write('Based on the input features, the model predicts:')
    if prediction == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    # Display the model accuracy
    st.header('Model Accuracy')
    st.write(f'Train set accuracy: {train_acc:.2f}')
    st.write(f'Test set accuracy: {test_acc:.2f}')

    # Developer Information
    st.title("About Developer")
    st.write("This app and ML model was developed by Gurjap Singh. The model uses Random Forest Classifier and is trained on a diabetes dataset.")
    image1 = Image.open(r"1729270232599.jpg")
    st.image(image1, width=200)
    st.write("Gurjap Singh (https://www.linkedin.com/in/gurjap-singh-46696332a/) age: 17 years as per 2024. I am a machine learning and AI enthusiast and developer")

if __name__ == '__main__':
    app()
