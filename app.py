import streamlit as st
import pickle
import numpy as np

diabetesr_fc_clf_scaled_model = pickle.load(open(r'D:\AMIT\amit\ODC\W3\D2\diabetes task\diabetes\diabetes.pkl', 'rb'))

def classify_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = diabetesr_fc_clf_scaled_model.predict(input_data_reshaped)
    prediction_prob = diabetesr_fc_clf_scaled_model.predict_proba(input_data_reshaped)[:,1]  
    return prediction, prediction_prob

def main():
    st.title("Diabetes Prediction App")
    st.write("This is a diabetes prediction app")

    pregnancies = st.slider("Pregnancies", 0, 17, 0)
    glucose = st.slider("Glucose", 126.0, 200.0, 99.0, step=0.1)
    blood_pressure = st.slider("Blood Pressure", 140.0, 180.0, 120.0, step=0.1)  
    skin_thickness = st.slider("Skin Thickness", 20.0, 99.0, 20.0, step=0.01) 
    insulin = st.slider("Insulin", 0, 900, 50)
    bmi = st.slider("BMI", 18.5, 40.0, 24.9, step=0.01)  
    diabetes_pedigree_function = st.slider("DiabetesPedigreeFunction", 0.0, 2.5, 0.5, step=0.01)
    age = st.slider("Age", 14.0, 100.0, 21.0, step=0.5)

    if st.button("Predict"):
        result, prediction_prob = classify_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
        
        if result == 1:
            st.markdown('<h3 style="color:red;">You have diabetes</h3>', unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:green;'>You don't have diabetes</h3>", unsafe_allow_html=True)
main()

