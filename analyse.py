import pandas as pd

# Define the file path
import pandas as pd

# Path to the CSV file
input_csv = r"D:\finalyearproject\Sleep-apnea-detection-through-a-modified-LeNet-5-convolutional-neural-network-master\output\sleep_data.csv"

# Read the CSV file using pandas
df = pd.read_csv(input_csv)

# Calculate the mean value for the 'Length minutes' column
mean_length_minutes = df['non-apn minutes'].sum()

# Print the mean value
print(f"Mean value for Length minutes: {mean_length_minutes}")


#///////////////////////////////////////////////////////////////

# import numpy as np
# import tensorflow as tf
# import os
# import pickle
# from scipy.interpolate import splev, splrep
# import matplotlib.pyplot as plt
# from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# # Base directory for the dataset
# base_dir = r"D:\finalyearproject\Sleep-apnea-detection-through-a-modified-LeNet-5-convolutional-neural-network-master\dataset\apnea-ecg-database-1.0.0"

# # Data scaling function
# scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# # Interpolation parameters
# ir = 3  # interpolate interval
# before = 2
# after = 2

# # Load and preprocess data
# def load_data():
#     tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

#     with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f:
#         apnea_ecg = pickle.load(f)

#     def process_data(data):
#         x_data = []
#         for record in data:
#             (rri_tm, rri_signal), (ampl_tm, ampl_signal) = record
#             rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
#             ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
#             x_data.append([rri_interp_signal, ampl_interp_signal])
#         x_data = np.array(x_data, dtype="float32").transpose((0, 2, 1))
#         return x_data

#     x_train = process_data(apnea_ecg["o_train"])
#     y_train = np.array(apnea_ecg["y_train"], dtype="float32")
#     groups_train = apnea_ecg["groups_train"]

#     x_test = process_data(apnea_ecg["o_test"])
#     y_test = np.array(apnea_ecg["y_test"], dtype="float32")
#     groups_test = apnea_ecg["groups_test"]

#     return (x_train, y_train, groups_train), (x_test, y_test, groups_test)

# # Inception module
# def inception_module(input_tensor, filters):
#     # Branch 1: 1x1 Convolution
#     branch1 = Conv1D(filters[0], kernel_size=1, activation='relu', padding='same')(input_tensor)

#     # Branch 2: 1x1 Convolution followed by 3x3 Convolution
#     branch2 = Conv1D(filters[1], kernel_size=1, activation='relu', padding='same')(input_tensor)
#     branch2 = Conv1D(filters[2], kernel_size=3, activation='relu', padding='same')(branch2)

#     # Branch 3: 1x1 Convolution followed by 5x5 Convolution
#     branch3 = Conv1D(filters[3], kernel_size=1, activation='relu', padding='same')(input_tensor)
#     branch3 = Conv1D(filters[4], kernel_size=5, activation='relu', padding='same')(branch3)

#     # Branch 4: MaxPooling followed by 1x1 Convolution
#     branch4 = MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
#     branch4 = Conv1D(filters[5], kernel_size=1, activation='relu', padding='same')(branch4)

#     # Concatenate all branches
#     output = concatenate([branch1, branch2, branch3, branch4], axis=-1)
#     return output

# # Build GoogleLeNet-1D model
# def build_googlelenet_1d(input_shape, num_classes):
#     inputs = Input(shape=input_shape)

#     # Initial Convolution and MaxPooling
#     x = Conv1D(64, kernel_size=7, strides=2, activation='relu', padding='same')(inputs)
#     x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

#     # Inception Modules
#     x = inception_module(x, [64, 96, 128, 16, 32, 32])
#     x = inception_module(x, [128, 128, 192, 32, 96, 64])
#     x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

#     x = inception_module(x, [192, 96, 208, 16, 48, 64])
#     x = inception_module(x, [160, 112, 224, 24, 64, 64])
#     x = inception_module(x, [128, 128, 256, 24, 64, 64])

#     # Global Average Pooling
#     x = GlobalAveragePooling1D()(x)

#     # Fully Connected Layers
#     x = Dense(512, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     outputs = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# # Plot training history
# def plot(history):
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # Plot loss
#     axes[0].plot(history["loss"], "b-", label="Training Loss")
#     axes[0].plot(history["val_loss"], "r-", label="Validation Loss")
#     axes[0].set_title("Loss")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Loss")
#     axes[0].legend()

#     # Plot accuracy
#     axes[1].plot(history["accuracy"], "b-", label="Training Accuracy")
#     axes[1].plot(history["val_accuracy"], "r-", label="Validation Accuracy")
#     axes[1].set_title("Accuracy")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("Accuracy")
#     axes[1].legend()

#     plt.tight_layout()
#     plt.savefig("performance_curvesGLN.png")
#     plt.show()

# # Main script
# if __name__ == "__main__":
#     # Load data
#     (x_train, y_train, groups_train), (x_test, y_test, groups_test) = load_data()

#     # Define input shape and number of classes
#     input_shape = (x_train.shape[1], x_train.shape[2])
#     num_classes = 2  # Apnea vs Non-Apnea

#     # Build model
#     model = build_googlelenet_1d(input_shape, num_classes)
#     model.summary()

#     # Compile model
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     # Train model
#     history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32)

#     # Evaluate model
#     loss, accuracy = model.evaluate(x_test, y_test)
#     print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

#     # Save model
#     model.save("modelgln.h5")

#     # Predictions
#     y_pred_prob = model.predict(x_test)
#     y_pred = np.argmax(y_pred_prob, axis=-1)

#     # Label mapping
#     class_mapping = {0: "Non-Apnea", 1: "Apnea"}

#     # Display predictions
#     for i, pred in enumerate(y_pred):
#         actual = class_mapping[y_test[i]]
#         predicted = class_mapping[pred]
#         print(f"Sample {i}: Actual = {actual}, Predicted = {predicted}")

#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:")
#     print(cm)

#     # Classification Report
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))

#     # AUC-ROC Score
#     roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
#     print(f"AUC-ROC Score: {roc_auc:.4f}")

#     # Plot performance
#     plot(history.history)

#////////////////////////////////////////////////////////////////////////////////////

# import numpy as np
# import tensorflow as tf
# import os
# import pickle
# from scipy.interpolate import splev, splrep
# import matplotlib.pyplot as plt
# from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# # Base directory for the dataset
# base_dir = r"D:\finalyearproject\Sleep-apnea-detection-through-a-modified-LeNet-5-convolutional-neural-network-master\dataset\apnea-ecg-database-1.0.0"

# # Data scaling function
# scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# # Interpolation parameters
# ir = 3  # interpolate interval
# before = 2
# after = 2

# # Load and preprocess data
# def load_data():
#     tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

#     with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f:
#         apnea_ecg = pickle.load(f)

#     def process_data(data):
#         x_data = []
#         for record in data:
#             (rri_tm, rri_signal), (ampl_tm, ampl_signal) = record
#             rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
#             ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
#             x_data.append([rri_interp_signal, ampl_interp_signal])
#         x_data = np.array(x_data, dtype="float32").transpose((0, 2, 1))
#         return x_data

#     x_train = process_data(apnea_ecg["o_train"])
#     y_train = np.array(apnea_ecg["y_train"], dtype="float32")
#     groups_train = apnea_ecg["groups_train"]

#     x_test = process_data(apnea_ecg["o_test"])
#     y_test = np.array(apnea_ecg["y_test"], dtype="float32")
#     groups_test = apnea_ecg["groups_test"]

#     return (x_train, y_train, groups_train), (x_test, y_test, groups_test)

# # Inception module
# def inception_module(input_tensor, filters):
#     # Branch 1: 1x1 Convolution
#     branch1 = Conv1D(filters[0], kernel_size=1, activation='relu', padding='same')(input_tensor)

#     # Branch 2: 1x1 Convolution followed by 3x3 Convolution
#     branch2 = Conv1D(filters[1], kernel_size=1, activation='relu', padding='same')(input_tensor)
#     branch2 = Conv1D(filters[2], kernel_size=3, activation='relu', padding='same')(branch2)

#     # Branch 3: 1x1 Convolution followed by 5x5 Convolution
#     branch3 = Conv1D(filters[3], kernel_size=1, activation='relu', padding='same')(input_tensor)
#     branch3 = Conv1D(filters[4], kernel_size=5, activation='relu', padding='same')(branch3)

#     # Branch 4: MaxPooling followed by 1x1 Convolution
#     branch4 = MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
#     branch4 = Conv1D(filters[5], kernel_size=1, activation='relu', padding='same')(branch4)

#     # Concatenate all branches
#     output = concatenate([branch1, branch2, branch3, branch4], axis=-1)
#     return output

# # Build GoogleLeNet-1D model
# def build_googlelenet_1d(input_shape, num_classes):
#     inputs = Input(shape=input_shape)

#     # Initial Convolution and MaxPooling
#     x = Conv1D(64, kernel_size=7, strides=2, activation='relu', padding='same')(inputs)
#     x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

#     # Inception Modules
#     x = inception_module(x, [64, 96, 128, 16, 32, 32])
#     x = inception_module(x, [128, 128, 192, 32, 96, 64])
#     x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

#     x = inception_module(x, [192, 96, 208, 16, 48, 64])
#     x = inception_module(x, [160, 112, 224, 24, 64, 64])
#     x = inception_module(x, [128, 128, 256, 24, 64, 64])

#     # Global Average Pooling
#     x = GlobalAveragePooling1D()(x)

#     # Fully Connected Layers
#     x = Dense(512, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     outputs = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# # Plot training history
# def plot(history):
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # Plot loss
#     axes[0].plot(history["loss"], "b-", label="Training Loss")
#     axes[0].plot(history["val_loss"], "r-", label="Validation Loss")
#     axes[0].set_title("Loss")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Loss")
#     axes[0].legend()

#     # Plot accuracy
#     axes[1].plot(history["accuracy"], "b-", label="Training Accuracy")
#     axes[1].plot(history["val_accuracy"], "r-", label="Validation Accuracy")
#     axes[1].set_title("Accuracy")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("Accuracy")
#     axes[1].legend()

#     plt.tight_layout()
#     plt.savefig("performance_curvesGLN.png")
#     plt.show()

# # Plot ECG with Apnea Regions
# def plot_ecg_with_apnea(x_data, y_data, sample_start, sample_end, class_mapping):
#     """
#     Plot ECG signals for a specific range with marked apnea regions.
    
#     Args:
#         x_data (np.ndarray): ECG signal data (2D array).
#         y_data (np.ndarray): Labels (0: Non-Apnea, 1: Apnea).
#         sample_start (int): Start index of the sample range.
#         sample_end (int): End index of the sample range (exclusive).
#         class_mapping (dict): Mapping of class indices to labels.
#     """
#     for i in range(sample_start, sample_end):
#         if i >= len(x_data):
#             break
        
#         # Extract the signals (rri_signal and ampl_signal)
#         rri_signal = x_data[i, 0, :]  # RRI signal for the i-th sample
#         ampl_signal = x_data[i, 1, :]  # Amplified ECG signal for the i-th sample
        
#         # Determine the label for the current sample
#         label = y_data[i]
#         label_str = class_mapping[label]
        
#         # Create a new plot for each sample
#         plt.figure(figsize=(10, 6))
        
#         # Plot the ECG signals
#         plt.plot(rri_signal, label='RRI Signal', color='blue', alpha=0.6)
#         plt.plot(ampl_signal, label='Amplified ECG Signal', color='green', alpha=0.6)
        
#         # Mark Apnea region if the label is 'Apnea'
#         if label == 1:
#             # Assuming the apnea period is a specific segment, we'll highlight the entire signal
#             plt.fill_between(range(len(ampl_signal)), min(ampl_signal), max(ampl_signal), color='red', alpha=0.3, label='Apnea Region')
        
#         # Plot title and labels
#         plt.title(f"ECG Signal Sample {i} - {label_str}")
#         plt.xlabel('Time (s)')
#         plt.ylabel('Signal Amplitude')
#         plt.legend()
#         plt.grid(True)
        
#         # Show the plot
#         plt.show()

# # Main script
# if __name__ == "__main__":
#     # Load data
#     (x_train, y_train, groups_train), (x_test, y_test, groups_test) = load_data()

#     # Define input shape and number of classes
#     input_shape = (x_train.shape[1], x_train.shape[2])
#     num_classes = 2  # Apnea vs Non-Apnea

#     # Build model
#     model = build_googlelenet_1d(input_shape, num_classes)
#     model.summary()

#     # Compile model
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     # Train model
#     history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32)

#     # Evaluate model
#     loss, accuracy = model.evaluate(x_test, y_test)
#     print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

#     # Save model
#     model.save("modelgln.h5")

#     # Predictions
#     y_pred_prob = model.predict(x_test)
#     y_pred = np.argmax(y_pred_prob, axis=-1)

#     # Label mapping
#     class_mapping = {0: "Non-Apnea", 1: "Apnea"}

#     # Display predictions
#     for i, pred in enumerate(y_pred):
#         actual = class_mapping[y_test[i]]
#         predicted = class_mapping[pred]
#         print(f"Sample {i}: Actual = {actual}, Predicted = {predicted}")

#     sample_start = int(input("Enter the starting sample number: "))
#     sample_end = int(input("Enter the ending sample number (exclusive): "))

#     # Plot ECG signals with marked apnea regions
#     plot_ecg_with_apnea(x_test, y_test, sample_start, sample_end, class_mapping)

#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:")
#     print(cm)

#     # Classification Report
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))

#     # AUC-ROC Score
#     roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
#     print(f"AUC-ROC Score: {roc_auc:.4f}")

#     # Plot performance
#     plot(history.history)

    # Get range of samples to display ECG with marked apnea regions

#/////////////////////////////////////////////////////////////////////

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from scipy.interpolate import splev, splrep

# # -------------------------------
# # ECG Data Preprocessing Function
# # -------------------------------

# def preprocess_ecg_data(raw_data, ir=3, before=2, after=2):
#     """
#     Preprocess raw ECG data to match the model input format.
#     Interpolates the RRI and Amplitude signals.

#     :param raw_data: List of tuples containing (rri, amplitude) time-series data
#     :param ir: Interpolation rate
#     :param before: Minutes before apnea event
#     :param after: Minutes after apnea event
#     :return: Preprocessed data with shape (samples, time_steps, 2)
#     """
#     scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
#     time_steps = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))
    
#     processed_data = []
#     for rri_data, ampl_data in raw_data:
#         rri_tm, rri_signal = rri_data
#         ampl_tm, ampl_signal = ampl_data

#         # Interpolate signals
#         rri_interp_signal = splev(time_steps, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
#         ampl_interp_signal = splev(time_steps, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)

#         # Combine both signals into a single sample
#         processed_data.append([rri_interp_signal, ampl_interp_signal])
    
#     processed_data = np.array(processed_data, dtype="float32").transpose((0, 2, 1))
#     return processed_data

# # -------------------
# # Synthetic ECG Data
# # -------------------

# def generate_synthetic_ecg(batch_size, time_steps, channels):
#     """
#     Generate synthetic ECG data for testing purposes.

#     :param batch_size: Number of samples in the batch
#     :param time_steps: Number of time steps (e.g., interpolated points)
#     :param channels: Number of features (e.g., RRI and Amplitude)
#     :return: Synthetic ECG data as a numpy array
#     """
#     np.random.seed(42)  # For reproducibility
#     synthetic_data = np.random.rand(batch_size, time_steps, channels)
#     return synthetic_data

# # ---------------------
# # Model Loading & Usage
# # ---------------------

# def load_and_test_model(model_path, input_data):
#     """
#     Load the trained model and make predictions on input data.

#     :param model_path: Path to the saved model file
#     :param input_data: Preprocessed input data (e.g., ECG signals)
#     :return: Predicted classes and probabilities
#     """
#     # Load the trained model
#     model = load_model(model_path)
#     print("Model loaded successfully.")

#     # Predict on the input data
#     predictions = model.predict(input_data)
#     predicted_classes = np.argmax(predictions, axis=-1)  # Get class index with the highest probability

#     return predicted_classes, predictions

# # ---------------
# # Main Execution
# # ---------------

# if __name__ == "__main__":
#     model_path = "modelgln.h5"
#     batch_size = 3  # Number of test samples
#     time_steps = 900  # Number of time steps (adjusted to match model input)
#     channels = 2  # Two features: RRI and Amplitude

#     # Generate synthetic ECG data
#     synthetic_ecg_data = generate_synthetic_ecg(batch_size, time_steps, channels)
#     print("Synthetic ECG data shape:", synthetic_ecg_data.shape)  # Ensure (3, 900, 2)

#     # Load model and test
#     predicted_classes, predictions = load_and_test_model(model_path, synthetic_ecg_data)

#     # Display predictions
#     print("Predicted Classes:", predicted_classes)
#     for i, probs in enumerate(predictions):
#         print(f"Sample {i + 1} Probabilities: {probs}")

# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model

# # Function to generate synthetic ECG data with apnea
# def generate_synthetic_ecg_with_apnea(batch_size, time_steps, channels):
#     """
#     Generate synthetic ECG data with simulated apnea-like abnormalities.
#     - Apnea is represented by irregular RRI and lower amplitude for a portion of the signal.
#     Args:
#         batch_size (int): Number of samples to generate.
#         time_steps (int): Number of time steps per sample.
#         channels (int): Number of features (e.g., RRI, amplitude).
#     Returns:
#         np.ndarray: Synthetic ECG data with apnea-like patterns.
#     """
#     np.random.seed(42)
#     synthetic_data = np.random.rand(batch_size, time_steps, channels)  # Random baseline data

#     for i in range(batch_size):
#         # Introduce apnea patterns in a portion of the signal
#         apnea_start = time_steps // 3
#         apnea_end = 2 * time_steps // 3

#         # Channel 0: RRI (introduce irregularity)
#         synthetic_data[i, apnea_start:apnea_end, 0] *= 1.5  # Stretch RRI to simulate irregular heartbeats

#         # Channel 1: Amplitude (reduce to simulate apnea event)
#         synthetic_data[i, apnea_start:apnea_end, 1] *= 0.5  # Lower amplitude to mimic apnea-like signal

#     return synthetic_data

# # Function to visualize synthetic ECG data
# def plot_synthetic_ecg(data, sample_idx):
#     """
#     Plot synthetic ECG data for a specific sample.
#     Args:
#         data (np.ndarray): Synthetic ECG data with shape (batch_size, time_steps, channels).
#         sample_idx (int): Index of the sample to plot.
#     """
#     rri = data[sample_idx, :, 0]
#     amplitude = data[sample_idx, :, 1]
#     time = np.arange(data.shape[1])

#     plt.figure(figsize=(12, 6))

#     # Plot RRI
#     plt.subplot(2, 1, 1)
#     plt.plot(time, rri, label="RRI", color="blue")
#     plt.axvspan(300, 600, color="red", alpha=0.2, label="Apnea Region")
#     plt.title("Synthetic RRI with Apnea")
#     plt.xlabel("Time Steps")
#     plt.ylabel("RRI")
#     plt.legend()

#     # Plot Amplitude
#     plt.subplot(2, 1, 2)
#     plt.plot(time, amplitude, label="Amplitude", color="green")
#     plt.axvspan(300, 600, color="red", alpha=0.2, label="Apnea Region")
#     plt.title("Synthetic Amplitude with Apnea")
#     plt.xlabel("Time Steps")
#     plt.ylabel("Amplitude")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# # Function to load the model and test it with input data
# def load_and_test_model(model_path, input_data):
#     """
#     Load a pre-trained model and make predictions on the given input data.
#     Args:
#         model_path (str): Path to the saved model.
#         input_data (np.ndarray): Input data for prediction.
#     Returns:
#         tuple: Predicted classes and probabilities.
#     """
#     model = load_model(model_path)
#     print("Model loaded successfully.")

#     # Make predictions
#     predictions = model.predict(input_data)
#     predicted_classes = np.argmax(predictions, axis=-1)

#     return predicted_classes, predictions

# # Main function
# if __name__ == "__main__":
#     # Parameters
#     batch_size = 3  # Number of test samples
#     time_steps = 900  # Number of time steps (same as the model's expected input)
#     channels = 2  # Two features: RRI and Amplitude
#     model_path = "modelgln.h5"  # Path to the saved model

#     # Generate synthetic ECG data with apnea
#     synthetic_ecg_apnea_data = generate_synthetic_ecg_with_apnea(batch_size, time_steps, channels)
#     # print("Shape of synthetic ECG data with apnea:", synthetic_ecg_apnea_data.shape)

#     # Visualize one sample
#     plot_synthetic_ecg(synthetic_ecg_apnea_data, sample_idx=0)

#     # Load model and make predictions
#     predicted_classes, predictions = load_and_test_model(model_path, synthetic_ecg_apnea_data)

#     # Display predictions
#     print("Shape of synthetic ECG data with apnea:", synthetic_ecg_apnea_data.shape)
#     print("Predicted Classes:", predicted_classes)
#     for i, probs in enumerate(predictions):
#         print(f"Sample {i + 1} Probabilities: {probs}")