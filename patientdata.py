import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from scipy.interpolate import splev, splrep
import wfdb

# Define paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(PROJECT_DIR, "models", "custom_lenet5_model.h5")
base_dir = os.path.join(PROJECT_DIR, "dataset", "apnea-ecg-database-1.0.0")

fs = 100  # Sampling frequency
segment_length = 900  # Length of each segment in samples

ir = 3
before = 2
after = 2

# Normalize function
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def load_person_data(person_id):
    tm = np.arange(0, segment_length / fs, step=1 / float(fs))
    
    # Load the raw data for the person
    signals, fields = wfdb.rdsamp(os.path.join(base_dir, person_id), channels=[0])
    signals = signals[:, 0]
    
    x_data = []
    for start in range(0, len(signals) - segment_length + 1, segment_length):
        segment = signals[start:start + segment_length]
        
        # Apply necessary preprocessing steps
        rri_interp_signal = splev(tm, splrep(tm, scaler(segment), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(tm, scaler(segment), k=3), ext=1)
        x_data.append([rri_interp_signal, ampl_interp_signal])
    
    x_data = np.array(x_data, dtype="float32").transpose((0, 2, 1))
    x_data = np.expand_dims(x_data, axis=-1)  # Add channel dimension
    
    return x_data, signals

def predict_sa(person_id):
    # Load the trained model
    model = load_model(model_path)
    
    # Load and preprocess the person's data
    x_data, signals = load_person_data(person_id)
    
    # Make predictions
    y_pred = model.predict(x_data)
    y_pred = np.argmax(y_pred, axis=-1)  # Convert probabilities to class labels
    
    # Calculate the number of apnea and hypopnea episodes
    apnea_episodes = np.sum(y_pred == 1)  # Adjust according to your model's classes
    hypopnea_episodes = np.sum(y_pred == 2)  # Adjust according to your model's classes
    
    print(f"Person ID: {person_id}")
    print(f"Apnea Episodes: {apnea_episodes}")
    print(f"Hypopnea Episodes: {hypopnea_episodes}")
    
    # Calculate total sleep time in minutes
    total_sleep_minutes = len(signals) / (fs * 60)  # Convert total samples to minutes
    print(f"Total Sleep Minutes: {total_sleep_minutes}")
    
    # Calculate AHI
    total_events = apnea_episodes + hypopnea_episodes
    ahi = (total_events / total_sleep_minutes) * 60
    print(f"Apnea-Hypopnea Index (AHI): {ahi:.2f}")
    
    # Determine SA status
    if apnea_episodes > hypopnea_episodes:
        print("Diagnosis: Positive for Sleep Apnea")
    else:
        print("Diagnosis: Negative for Sleep Apnea")

# Example usage
person_id = "x23"  # Replace with the actual person ID
predict_sa(person_id)
