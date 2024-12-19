import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from PIL import Image

# load the diabetes dataset
diabetes_df = pd.read_csv('diabetes (2).csv')

# group the data by outcome to get a sense of the distribution
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# split the data into input and target variables
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create the RandomForestClassifier model
model = RandomForestClassifier(random_state=1)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# train the model on the training set
best_model.fit(X_train, y_train)

# make predictions on the training and testing sets
train_y_pred = best_model.predict(X_train)
test_y_pred = best_model.predict(X_test)

# calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)

# create the Streamlit app
def app():
    st.title("Predict Diabetes")

    img = Image.open(r"img.jpeg")
    img = img.resize((200, 200))
    st.image(img, width=200)

    st.title('Diabetes Prediction')

    # create the input form for the user to input new data
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # make a prediction based on the user input
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    input_data_nparray = np.asarray(input_data)
    reshaped_input_data = input_data_nparray.reshape(1, -1)
    prediction = best_model.predict(reshaped_input_data)

    # display the prediction to the user
    st.write('Based on the input features, the model predicts:')
    if prediction == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    # Display model evaluation metrics
    st.header('Model Evaluation')
    st.write(f'Train set accuracy: {train_acc:.2f}')
    st.write(f'Test set accuracy: {test_acc:.2f}')
    
    st.subheader('Classification Report')
    st.text(classification_report(y_test, test_y_pred))

    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, test_y_pred)
    st.write(cm)

    # display the feature importance
    st.header('Feature Importance')
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.write(feature_importance)

    st.title("About Developer")
    st.write("This app and ML model was developed by Gurjap Singh, trained on a diabetes dataset. The model uses Random Forest Classifier and is trained on a 75%-25% split of the dataset.")
    image1 = Image.open(r"1729270232599.jpg")
    st.image(image1, width=200)
    st.write("Gurjap Singh (https://www.linkedin.com/in/gurjap-singh-46696332a/) age: 17 years as of 2024. I am a machine learning and AI enthusiast and developer.")

if __name__ == '__main__':
    app()
