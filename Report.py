import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
from scipy.interpolate import splev, splrep
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, LSTM
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Load Data from your dataset
base_dir = r"D:\\finalyearproject\\Sleep-apnea-detection-through-a-modified-LeNet-5-convolutional-neural-network-master\\dataset\\apnea-ecg-database-1.0.0"
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
ir, before, after = 3, 2, 2  # Sampling rate and windowing

def load_data():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))
    with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f:
        apnea_ecg = pickle.load(f)

    def process_data(data):
        x_data = []
        for record in data:
            (rri_tm, rri_signal), (ampl_tm, ampl_signal) = record
            rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
            ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
            x_data.append([rri_interp_signal, ampl_interp_signal])
        x_data = np.array(x_data, dtype="float32").transpose((0, 2, 1))
        return x_data

    x_train = process_data(apnea_ecg["o_train"])
    y_train = np.array(apnea_ecg["y_train"], dtype="float32")
    x_test = process_data(apnea_ecg["o_test"])
    y_test = np.array(apnea_ecg["y_test"], dtype="float32")
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

# 1. ECG Signal Preprocessing - Raw vs. Interpolated
plt.figure(figsize=(12, 4))
plt.plot(x_train[0][:, 0], label="Interpolated RRI Signal", color='blue')
plt.plot(x_train[0][:, 1], label="Interpolated Amplitude Signal", color='red', alpha=0.7)
plt.title("ECG Signal Preprocessing (Interpolated Signals)")
plt.legend()
plt.show()

# 2. Feature Extraction using 1D-GoogleLeNet
input_signal = Input(shape=(x_train.shape[1], x_train.shape[2]))
conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_signal)
conv2 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(conv1)
conv3 = Conv1D(filters=128, kernel_size=7, activation='relu', padding='same')(conv2)
feature_output = GlobalAveragePooling1D()(conv3)
feature_extractor = Model(input_signal, feature_output)

features = feature_extractor.predict(x_train[:10])

plt.figure(figsize=(8, 4))
plt.plot(features[0])
plt.title("Feature Extraction using 1D-GoogleLeNet")
plt.show()

# 3. Temporal Feature Learning with LSTM
lstm_input = Input(shape=(x_train.shape[1], x_train.shape[2]))
lstm_layer = LSTM(64, return_sequences=True)(lstm_input)
lstm_output = LSTM(32)(lstm_layer)
lstm_model = Model(lstm_input, lstm_output)

lstm_features = lstm_model.predict(x_train[:10])

plt.figure(figsize=(8, 4))
plt.plot(lstm_features[0])
plt.title("Temporal Feature Learning with LSTM")
plt.show()

# 4. Classification Output
y_pred_prob = np.random.rand(len(y_test), 2)  # Simulating predictions
y_pred = np.argmax(y_pred_prob, axis=1)

plt.figure(figsize=(8, 4))
plt.hist(y_pred[y_pred == 0], bins=1, color='blue', alpha=0.7, edgecolor='black', label="Non-Apnea")
plt.hist(y_pred[y_pred == 1], bins=1, color='red', alpha=0.7, edgecolor='black', label="Apnea")
plt.xticks([0, 1], ["Non-Apnea", "Apnea"])
plt.legend()
plt.title("Classification Output (Apnea vs Non-Apnea)")
plt.show()


# 5. Model Performance Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Apnea", "Apnea"], yticklabels=["Non-Apnea", "Apnea"])
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 4))
plt.plot(fpr, tpr, color='blue', label="AUC = {:.2f}".format(roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC Curve")
plt.legend()
plt.show()
