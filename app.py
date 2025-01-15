import streamlit as st
import pickle
import numpy as np

logmodel = pickle.load(open('log_model.pkl', 'rb'))
def classify_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, BMI, Age):
    input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, BMI, Age)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = logmodel.predict(input_data_reshaped)
    return prediction
def main():
    st.title("Diabetes Prediction App")
    st.write("This is a diabetes prediction app")
    pregnancies = st.slider("Pregnancies", 0, 17, 3)
    glucose = st.slider("Glucose", 0, 199, 117)
    blood_pressure = st.slider("Blood Pressure", 0, 122, 72)
    skin_thickness = st.slider("Skin Thickness", 0, 99, 23)
    bmi = st.slider("BMI", 0.0, 67.1, 32.0)
    age = st.slider("Age", 21, 81, 29)
    if st.button("Predict"):
        result = classify_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, bmi, age)
        if result == 1:
            st.write("You have diabetes")
        else:
            st.write("You do not have diabetes")

main()


