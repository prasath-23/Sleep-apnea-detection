# import numpy as np
# import os
# import pickle
# import pandas as pd

# # Set base directory
# base_dir = r"D:\finalyearproject\Sleep-apnea-detection-through-a-modified-LeNet-5-convolutional-neural-network-master\dataset\apnea-ecg-database-1.0.0"

# # Load the dataset
# with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f:
#     apnea_ecg = pickle.load(f)

# # Function to extract and flatten data into rows
# def extract_records(data, label):
#     records = []
#     for idx, record in enumerate(data):
#         (rri_tm_values, rri_signal_values), (ampl_tm_values, ampl_signal_values) = record
#         for tm, rri, ampl_tm, ampl in zip(rri_tm_values, rri_signal_values, ampl_tm_values, ampl_signal_values):
#             records.append({
#                 "Record_ID": idx,
#                 "Time_RRI": tm,
#                 "Signal_RRI": rri,
#                 "Time_Amplitude": ampl_tm,
#                 "Signal_Amplitude": ampl,
#                 "Label": label
#             })
#     return records

# # Extract data for training and testing sets
# train_records = extract_records(apnea_ecg["o_train"], label="Train")
# test_records = extract_records(apnea_ecg["o_test"], label="Test")

# # Combine all records
# all_records = train_records + test_records

# # Convert to DataFrame
# df = pd.DataFrame(all_records)

# # Save to CSV
# output_csv = "all_records.csv"
# df.to_csv(output_csv, index=False)

# print(f"Data saved to {output_csv}")

import numpy as np
import pickle
import os
import pandas as pd

# Set base directory
base_dir = r"D:\finalyearproject\Sleep-apnea-detection-through-a-modified-LeNet-5-convolutional-neural-network-master\dataset\apnea-ecg-database-1.0.0"

# Load the apnea dataset
def load_apnea_dataset():
    """
    Load the apnea dataset from a pickle file.

    Returns:
        dict: Dictionary containing the apnea dataset.
    """
    with open(os.path.join(base_dir, "apnea-ecg.pkl"), "rb") as f:
        apnea_ecg = pickle.load(f)
    return apnea_ecg

# Save preprocessed data to CSV
def save_preprocessed_data_to_csv(apnea_ecg, output_csv_path):
    """
    Save the preprocessed RRI and amplitude data to a CSV file.

    Args:
        apnea_ecg (dict): Apnea dataset containing training and testing data.
        output_csv_path (str): Path to save the output CSV file.
    """
    records = []
    
    # Combine training and testing data
    for record in apnea_ecg["o_train"] + apnea_ecg["o_test"]:
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = record
        
        # Convert NumPy arrays to comma-separated strings
        rri_tm_str = np.array2string(rri_tm, separator=",", precision=5, suppress_small=True).replace("\n", "").strip("[]")
        rri_signal_str = np.array2string(rri_signal, separator=",", precision=5, suppress_small=True).replace("\n", "").strip("[]")
        ampl_tm_str = np.array2string(ampl_tm, separator=",", precision=5, suppress_small=True).replace("\n", "").strip("[]")
        ampl_signal_str = np.array2string(ampl_signal, separator=",", precision=5, suppress_small=True).replace("\n", "").strip("[]")
        
        # Append data as a single record
        records.append({
            "rri_tm": f"np.array([{rri_tm_str}])",
            "rri_signal": f"np.array([{rri_signal_str}])",
            "ampl_tm": f"np.array([{ampl_tm_str}])",
            "ampl_signal": f"np.array([{ampl_signal_str}])"
        })
    
    # Save the records to a CSV file
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Data successfully saved to {output_csv_path}")

if __name__ == "__main__":
    # Load the apnea dataset
    print("Loading apnea dataset...")
    apnea_ecg = load_apnea_dataset()

    # Specify output CSV file path
    output_csv_path = "preprocessed_apnea_data.csv"

    # Save the preprocessed data to CSV
    print("Saving preprocessed data to CSV...")
    save_preprocessed_data_to_csv(apnea_ecg, output_csv_path)
    print("Process completed successfully.")
