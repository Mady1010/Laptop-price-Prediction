import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler

def load_model():
    with open('regression2.pkl', 'rb') as file:
        gbr, scaler = pickle.load(file)
    return gbr, scaler

gbr, scaler = load_model()

# Function for prediction
def predict(inputs):
    input_features_scaled = scaler.transform(inputs)
    prediction = gbr.predict(input_features_scaled)
    return np.round(prediction)

# Streamlit UI
st.title("Laptop Price Prediction")

# Collecting user input
device_type = st.selectbox('Device Type', ['Notebook', 'Ultrabook'])
brand = st.selectbox('Brand', ['Dell', 'Apple', 'HP', 'Lenovo', 'Acer'])
storage_type = st.selectbox('Storage Type', ['SSD', 'HDD'])
os_type = st.selectbox('Operating System', ['Windows', 'Linux', 'DOS', 'MacOS'])
cpu_model = st.text_input('CPU Model')
gpu_memory = st.selectbox('GPU Memory', [10, 14, 18, 30])
gpu_brand = st.selectbox('GPU Brand', ['NVIDIA', 'AMD', 'Intel'])
ram = st.number_input('RAM (in GB)', min_value=0.0, max_value=64.0, step=4.0)
storage = st.number_input('Storage (in GB)', min_value=128, max_value=2048, step=128)
resolution = st.text_input('Resolution (Width x Height)', '1920x1080')

# Parse resolution
if resolution:
    resolution_width, resolution_height = map(int, resolution.split('x'))

# Initialize features dictionary based on user input
features = {
    "SSD": 1.0 if storage_type.lower() == "ssd" else 0.0,
    "18_GPU": 1.0 if gpu_memory == 18 else 0.0,
    "_DELL": 1.0 if brand.lower() == "dell" else 0.0,
    "_HDD": 1.0 if storage_type.lower() == "hdd" else 0.0,
    "AMD": 1.0 if "amd" in cpu_model.lower() else 0.0,
    "Intel": 1.0 if "intel" in cpu_model.lower() else 0.0,
    "_Dos": 1.0 if os_type.lower() == "dos" else 0.0,
    "AMD_GPU": 1.0 if "amd" in gpu_brand.lower() else 0.0,
    "14_GPU": 1.0 if gpu_memory == 14 else 0.0,
    "10_GPU": 1.0 if gpu_memory == 10 else 0.0,
    "NVIDIA_GPU": 1.0 if "nvidia" in gpu_brand.lower() else 0.0,
    "_Notebook": 1.0 if device_type.lower() == "notebook" else 0.0,
    "Intel_GPU": 1.0 if "intel" in gpu_brand.lower() else 0.0,
    "30_GPU": 1.0 if gpu_memory == 30 else 0.0,
    "_Apple": 1.0 if brand.lower() == "apple" else 0.0,
    "_Mac": 1.0 if "mac" in os_type.lower() else 0.0,
    "_Ultrabook": 1.0 if device_type.lower() == "ultrabook" else 0.0,
    "M3": 1.0 if cpu_model.lower() == "m3" else 0.0,
    "ResolutionWidth": resolution_width,
    "ResolutionHeight": resolution_height,
    "Memory Amount": storage * 1000 if storage in ['512', '256'] else storage * 100000,  # Convert GB to MB
    "RAM": ram
}

# Convert features to an array for prediction
input_array = np.array(list(features.values())).reshape(1, -1)

# Button to trigger prediction
if st.button('Predict Price'):
    # Get prediction
    prediction = predict(input_array)
    st.write(f"The predicted laptop price is: ${prediction[0]}")

