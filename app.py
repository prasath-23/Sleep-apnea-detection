from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os
import pickle
import io
import base64
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

app = Flask(__name__)
CORS(app)

# Path to the dataset
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(PROJECT_DIR, "dataset", "apnea-ecg-database-1.0.0")

# Global variables for metrics
loss, accuracy = None, None
y_test, y_pred_prob, y_pred = None, None, None

# Data loading and preprocessing
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
ir, before, after = 3, 2, 2

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

@app.route('/metrics', methods=['GET'])
def get_metrics():
    global loss, accuracy, y_test, y_pred_prob, y_pred
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    roc_curve_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return jsonify({
        'loss': loss,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'roc_curve_image': roc_curve_image
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict Apnea or Non-Apnea based on user input signals.
    Expects JSON input with rri_tm, rri_signal, ampl_tm, ampl_signal.
    """
    data = request.get_json()
    try:
        rri_tm = np.array(data['rri_tm'], dtype="float32")
        rri_signal = np.array(data['rri_signal'], dtype="float32")
        ampl_tm = np.array(data['ampl_tm'], dtype="float32")
        ampl_signal = np.array(data['ampl_signal'], dtype="float32")
        
        # Ensure sufficient points for interpolation
        if len(rri_tm) < 4 or len(ampl_tm) < 4:
            return jsonify({'error': 'Insufficient points for interpolation. Provide at least 4 points per input.'}), 400

        # Preprocess the input
        tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        single_input_processed = np.array([[rri_interp_signal, ampl_interp_signal]], dtype="float32").transpose((0, 2, 1))
        
        # Predict using the trained model
        prediction_prob = model.predict(single_input_processed, verbose=0)
        predicted_label = np.argmax(prediction_prob, axis=-1)[0]
        apnea_prob = prediction_prob[0][1]
        prediction = "Apnea" if predicted_label == 1 else "Non-Apnea"

        # Convert float32 values to standard float for JSON serialization
        return jsonify({
            'prediction': prediction,
            'apnea_probability': float(apnea_prob)  # Convert to standard float
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_points', methods=['POST'])
def detect_points():
    """
    Detect specific time points where OSA is predicted based on user input signals.
    Expects JSON input with rri_tm, rri_signal, ampl_tm, ampl_signal.
    """
    data = request.get_json()
    try:
        rri_tm = np.array(data['rri_tm'], dtype="float32")
        rri_signal = np.array(data['rri_signal'], dtype="float32")
        ampl_tm = np.array(data['ampl_tm'], dtype="float32")
        ampl_signal = np.array(data['ampl_signal'], dtype="float32")
        
        # Ensure sufficient points for interpolation
        if len(rri_tm) < 4 or len(ampl_tm) < 4:
            return jsonify({'error': 'Insufficient points for interpolation. Provide at least 4 points per input.'}), 400

        # Preprocess the entire input
        tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)

        # Define the window size and step (e.g., 30 seconds)
        window_size = int(30 * ir)  # Convert seconds to sample points
        step_size = int(10 * ir)  # Sliding step size (e.g., 10 seconds)
        detected_points = []

        # Slide through the signal with the defined window
        for start in range(0, len(tm) - window_size + 1, step_size):
            end = start + window_size
            segment_rri = rri_interp_signal[start:end]
            segment_ampl = ampl_interp_signal[start:end]
            if len(segment_rri) < window_size or len(segment_ampl) < window_size:
                continue
            
            # Preprocess segment for model input
            segment_input = np.array([[segment_rri, segment_ampl]], dtype="float32").transpose((0, 2, 1))
            
            # Predict on the segment
            prediction_prob = model.predict(segment_input, verbose=0)
            predicted_label = np.argmax(prediction_prob, axis=-1)[0]
            
            if predicted_label == 1:  # If "Apnea" is detected
                detected_points.append({
                    'start_time': tm[start],
                    'end_time': tm[end - 1],
                    'apnea_probability': float(prediction_prob[0][1])
                })

        return jsonify({
            'detected_points': detected_points,
            'message': 'Detection completed successfully!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    input_shape = (x_train.shape[1], x_train.shape[2])
    num_classes = 2
    model = build_googlelenet_1d(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
    loss, accuracy = model.evaluate(x_test, y_test)
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=-1)
    app.run(debug=True)