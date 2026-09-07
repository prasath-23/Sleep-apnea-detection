import pickle
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import netron
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Input
from keras.models import Model
from keras.regularizers import l2
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(PROJECT_DIR, "dataset", "apnea-ecg-database-1.0.0")

ir = 3  # interpolate interval
before = 2
after = 2

# normalize
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def load_data():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f:
        apnea_ecg = pickle.load(f)

    x_train = []
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_train[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x_train.append([rri_interp_signal, ampl_interp_signal])
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1))
    y_train = np.array(y_train, dtype="float32")

    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_test[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x_test.append([rri_interp_signal, ampl_interp_signal])
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    return x_train, y_train, groups_train, x_test, y_test, groups_test

def create_model(input_shape, weight=1e-3):
    """Create a Modified LeNet-5 model"""
    inputs = Input(shape=input_shape)

    # Conv1
    x = Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x = MaxPooling1D(pool_size=3)(x)

    # Conv3
    x = Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = MaxPooling1D(pool_size=3)(x)

    x = Dropout(0.8)(x)  # Avoid overfitting

    # FC6
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def lr_schedule(epoch, lr):
    if epoch > 70 and (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr

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
    fig.savefig(os.path.join(PROJECT_DIR, "performance_curvesle.png"))
    plt.show()

if __name__ == "__main__":
    x_train, y_train, groups_train, x_test, y_test, groups_test = load_data()

    # One-hot encoding the labels
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    print("train num:", len(y_train))
    print("test num:", len(y_test))

    model = create_model(input_shape=x_train.shape[1:])
    model.summary()

    # Save and visualize the model
    model_preview_path = os.path.join(PROJECT_DIR, "modelnew.h5")
    model.save(model_preview_path)
    netron.start(model_preview_path)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    lr_scheduler = LearningRateScheduler(lr_schedule)
    history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test),
                        callbacks=[lr_scheduler])
    model.save(os.path.join(PROJECT_DIR, "models", "model.final.h5"))

    # Evaluate model on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Make predictions
    y_pred = model.predict(x_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_test_class = np.argmax(y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_test_class, y_pred_class)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    class_report = classification_report(y_test_class, y_pred_class)
    print("Classification Report:")
    print(class_report)

    # Compute AUC-ROC
    y_test_binary = y_test[:, 1]  # Extract true class probabilities
    y_pred_binary = y_pred[:, 1]  # Extract predicted probabilities for class 1
    roc_auc = roc_auc_score(y_test_binary, y_pred_binary)
    print(f"AUC-ROC Score: {roc_auc:.4f}")

    # Compute ROC Curve
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Save results
    output_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    output = pd.DataFrame({"y_true": y_test_binary, "y_score": y_pred_binary, "subject": groups_test})
    output.to_csv(os.path.join(output_dir, "LeNet.csv"), index=False)

    # Plot training history
    plot(history.history)
