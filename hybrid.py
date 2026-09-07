import numpy as np
import tensorflow as tf
import os
import pickle
import netron
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, LSTM, concatenate, BatchNormalization, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(PROJECT_DIR, "dataset", "apnea-ecg-database-1.0.0")

scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

ir = 3  # interpolate interval
before = 2
after = 2

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
    groups_train = apnea_ecg["groups_train"]

    x_test = process_data(apnea_ecg["o_test"])
    y_test = np.array(apnea_ecg["y_test"], dtype="float32")
    groups_test = apnea_ecg["groups_test"]

    return (x_train, y_train, groups_train), (x_test, y_test, groups_test)

def inception_module(input_tensor, filters):
    branch1 = Conv1D(filters[0], kernel_size=1, activation='relu', padding='same')(input_tensor)
    branch1 = BatchNormalization()(branch1)

    branch2 = Conv1D(filters[1], kernel_size=1, activation='relu', padding='same')(input_tensor)
    branch2 = Conv1D(filters[2], kernel_size=3, activation='relu', padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)

    branch3 = Conv1D(filters[3], kernel_size=1, activation='relu', padding='same')(input_tensor)
    branch3 = Conv1D(filters[4], kernel_size=5, activation='relu', padding='same')(branch3)
    branch3 = BatchNormalization()(branch3)

    branch4 = MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
    branch4 = Conv1D(filters[5], kernel_size=1, activation='relu', padding='same')(branch4)
    branch4 = BatchNormalization()(branch4)

    output = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    return output

def build_hybrid_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # 1D-GoogleLeNet feature extractor
    x = Conv1D(64, kernel_size=7, strides=2, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = inception_module(x, [64, 96, 128, 16, 32, 32])
    x = inception_module(x, [128, 128, 192, 32, 96, 64])
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = inception_module(x, [192, 96, 208, 16, 48, 64])
    x = inception_module(x, [160, 112, 224, 24, 64, 64])
    x = inception_module(x, [128, 128, 256, 24, 64, 64])
    x = GlobalAveragePooling1D()(x)

    # Reshape for LSTM using Lambda layer
    x = Lambda(lambda t: tf.expand_dims(t, axis=1))(x)

    # LSTM layers
    x = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
    x = LSTM(64, dropout=0.3, recurrent_dropout=0.3)(x)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def plot(history):
    """Plot performance curves for training and validation"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting loss
    axes[0].plot(history["loss"], "b-", label="Training Loss", linewidth=1)
    axes[0].plot(history["val_loss"], "r-", label="Validation Loss", linewidth=1)
    axes[0].set_title("Loss", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend(loc="best")

    # Plotting accuracy
    axes[1].plot(history["accuracy"], "b-", label="Training Accuracy", linewidth=1)
    axes[1].plot(history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=1)
    axes[1].set_title("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].legend(loc="best")

    fig.tight_layout()

    fig.savefig(os.path.join(PROJECT_DIR, "performance_curveshyb.png"))
    
    plt.show()

if __name__ == "__main__":
    (x_train, y_train, groups_train), (x_test, y_test, groups_test) = load_data()

    input_shape = (x_train.shape[1], x_train.shape[2])
    num_classes = 2  # Apnea vs Non-Apnea

    # Build and compile the hybrid model
    model = build_hybrid_model(input_shape, num_classes)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_preview_path = os.path.join(PROJECT_DIR, "modelhyb.h5")
    model.save(model_preview_path)
    netron.start(model_preview_path)
    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    # checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)
    checkpoint = ModelCheckpoint(os.path.join(PROJECT_DIR, "best_model.keras"), monitor='val_accuracy', save_best_only=True)

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=1,
        batch_size=32,
        callbacks=[early_stopping, checkpoint]
    )
    
    # Load the best model
    # model.load_weights("best_model.h5")
    model.load_weights(os.path.join(PROJECT_DIR, "best_model.keras"))


    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Predictions and metrics
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    # plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    print("AUC-ROC Score:", roc_auc_score(y_test, y_pred))
    plot(history.history)