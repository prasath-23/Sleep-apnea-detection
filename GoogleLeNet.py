import numpy as np
import tensorflow as tf
import os
import pickle
import netron
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def scale_data(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

ir = 3  # interpolate interval
before = 2
after = 2

base_dir = os.path.join(PROJECT_DIR, "dataset", "apnea-ecg-database-1.0.0")

def load_data():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))
    with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f:
        apnea_ecg = pickle.load(f)
    
    def process_data(data):
        x_data = []
        for record in data:
            (rri_tm, rri_signal), (ampl_tm, ampl_signal) = record
            rri_interp_signal = splev(tm, splrep(rri_tm, scale_data(rri_signal), k=3), ext=1)
            ampl_interp_signal = splev(tm, splrep(ampl_tm, scale_data(ampl_signal), k=3), ext=1)
            x_data.append([rri_interp_signal, ampl_interp_signal])
        return np.array(x_data, dtype="float32").transpose((0, 2, 1))

    x_train = process_data(apnea_ecg["o_train"])
    y_train = np.array(apnea_ecg["y_train"], dtype="float32")
    x_test = process_data(apnea_ecg["o_test"])
    y_test = np.array(apnea_ecg["y_test"], dtype="float32")
    return (x_train, y_train), (x_test, y_test)

def inception_module(input_tensor, filters):
    branch1 = Conv1D(filters[0], kernel_size=1, activation='relu', padding='same')(input_tensor)
    branch2 = Conv1D(filters[1], kernel_size=1, activation='relu', padding='same')(input_tensor)
    branch2 = Conv1D(filters[2], kernel_size=3, activation='relu', padding='same')(branch2)
    branch3 = Conv1D(filters[3], kernel_size=1, activation='relu', padding='same')(input_tensor)
    branch3 = Conv1D(filters[4], kernel_size=5, activation='relu', padding='same')(branch3)
    branch4 = MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
    branch4 = Conv1D(filters[5], kernel_size=1, activation='relu', padding='same')(branch4)
    return concatenate([branch1, branch2, branch3, branch4], axis=-1)

def build_googlelenet_1d(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=7, strides=2, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    x = inception_module(x, [64, 96, 128, 16, 32, 32])
    x = inception_module(x, [128, 128, 192, 32, 96, 64])
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    x = inception_module(x, [192, 96, 208, 16, 48, 64])
    x = inception_module(x, [160, 112, 224, 24, 64, 64])
    x = inception_module(x, [128, 128, 256, 24, 64, 64])
    x = GlobalAveragePooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"AUC-ROC Score: {roc_auc:.4f}")

def plot_loss_accuracy(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.history["loss"], "b-", label="Training Loss", linewidth=1)
    axes[0].plot(history.history["val_loss"], "r-", label="Validation Loss", linewidth=1)
    axes[0].set_title("Loss", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend()
    
    axes[1].plot(history.history["accuracy"], "b-", label="Training Accuracy", linewidth=1)
    axes[1].plot(history.history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=1)
    axes[1].set_title("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].legend()
    
    plt.show()

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_googlelenet_1d(x_train.shape[1:], num_classes=2)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32)
    
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=-1)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    plot_loss_accuracy(history)
    plot_roc_curve(y_test, y_pred_prob)
