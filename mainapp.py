import streamlit as st
import pickle
import numpy as np

# Load pre-trained models (assuming they are saved in the same directory)
with open('heart_disease_model.sav', 'rb') as f:
    heart_disease_model = pickle.load(f)

with open('parkinson_model.sav', 'rb') as f:
    parkinson_model = pickle.load(f)

with open('diabetes_model.sav', 'rb') as f:
    diabetes_model = pickle.load(f)

# Sidebar with model selection
st.sidebar.title("Select a Model")
model = st.sidebar.selectbox("Choose a model to make predictions", 
                             ["Heart Disease", "Parkinson's Disease", "Diabetes"])
st.header("HEALTHCARE SYSTEM")
# Function to get inputs and make predictions for Heart Disease
def classify_disease_level(probability):
    if probability < 0.4:
        return "Low Risk"
    elif 0.4 <= probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"
    
def heart_disease_inputs():
    st.subheader("Heart Disease Model Inputs")
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])  # Integer encoding for categorical features
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
    chol = st.number_input("Cholesterol", min_value=100, max_value=400)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("Resting ECG Results", options=[0, 1])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0)
    slope = st.selectbox("Slope of the ST Segment", options=[0, 1, 2])  
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)  
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])  
    
    
    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex == 'Male', cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_disease_model.predict(input_data)
        # st.write("Prediction (0 = No Heart Disease, 1 = Heart Disease): ", prediction[0])
        if prediction[0] == 0:
            st.write("No Heart Disease , You are fit")
        else:
            st.write("Heart Disease , You are not fit")
            test_probabilities = heart_disease_model.predict_proba(input_data)[:, 1] 
            test_disease_levels = [classify_disease_level(prob) for prob in test_probabilities]
            st.write("Disease Level: ", test_disease_levels[0])


# Function to get inputs and make predictions for Parkinson's Disease
# Function to get inputs and make predictions for Parkinson's Disease
def parkinson_inputs():
    st.subheader("Parkinson's Disease Model Inputs")
    mdvp_fo = st.number_input("MDVP:Fo(Hz)", min_value=100.0, max_value=300.0)
    mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", min_value=100.0, max_value=400.0)
    mdvp_flo = st.number_input("MDVP:Flo(Hz)", min_value=50.0, max_value=250.0)
    jitter_percent = st.number_input("Jitter(%)", min_value=0.0, max_value=1.0)
    jitter_abs = st.number_input("Jitter(Abs)", min_value=0.0, max_value=0.1)
    rap = st.number_input("RAP", min_value=0.0, max_value=0.1)
    ppq = st.number_input("PPQ", min_value=0.0, max_value=0.1)
    ddp = st.number_input("DDP", min_value=0.0, max_value=0.3)
    mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=0.1)
    shimmer_db = st.number_input("Shimmer(dB)", min_value=0.0, max_value=2.0)
    apq3 = st.number_input("APQ3", min_value=0.0, max_value=0.1)
    apq5 = st.number_input("APQ5", min_value=0.0, max_value=0.2)
    apq = st.number_input("APQ", min_value=0.0, max_value=0.3)
    dda = st.number_input("DDA", min_value=0.0, max_value=0.3)
    nhr = st.number_input("NHR", min_value=0.0, max_value=1.0)
    hnr = st.number_input("HNR", min_value=0.0, max_value=40.0)
    rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0)
    dfa = st.number_input("DFA", min_value=0.0, max_value=1.0)
    spread1 = st.number_input("Spread1", min_value=-10.0, max_value=0.0)
    spread2 = st.number_input("Spread2", min_value=0.0, max_value=1.0)
    d2 = st.number_input("D2", min_value=0.0, max_value=4.0)
    ppe = st.number_input("PPE", min_value=0.0, max_value=1.0)

    if st.button("Predict Parkinson's Disease"):
        input_data = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, jitter_percent, jitter_abs, rap, ppq, ddp, mdvp_shimmer,
                                shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        prediction = parkinson_model.predict(input_data)
        # st.write("Prediction (0 = No Parkinson's Disease, 1 = Parkinson's Disease): ", prediction[0])
        if prediction[0] == 0:
            st.write("No Parkinson's Disease , You are fit")
        else:
            st.write("Parkinson's Disease , You are not fit")
            test_probabilities = parkinson_model.predict_proba(input_data)[:, 1] 
            test_disease_levels = [classify_disease_level(prob) for prob in test_probabilities]
            st.write("Disease Level: ", test_disease_levels[0])


# Function to get inputs and make predictions for Diabetes
def diabetes_inputs():
    st.subheader("Diabetes Model Inputs")
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=846)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
    age = st.number_input("Age", min_value=1, max_value=120)
    
    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(input_data)
        # st.write("Prediction (0 = No Diabetes, 1 = Diabetes): ", prediction[0])
        if prediction[0] == 0:
            st.write("No Diabetes , You are fit")
        else:
            st.write("Diabetes , You are not fit")
            test_probabilities = diabetes_model.predict_proba(input_data)[:, 1] 
            test_disease_levels = [classify_disease_level(prob) for prob in test_probabilities]
            st.write("Disease Level: ", test_disease_levels[0])

# Main area to show the form based on the model selected
if model == "Heart Disease":
    heart_disease_inputs()
elif model == "Parkinson's Disease":
    parkinson_inputs()
elif model == "Diabetes":
    diabetes_inputs()
