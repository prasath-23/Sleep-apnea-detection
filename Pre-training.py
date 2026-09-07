import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix

base_dir = r"D:\finalyearproject\Sleep-apnea-detection-through-a-modified-LeNet-5-convolutional-neural-network-master\dataset\apnea-ecg-database-1.0.0"

ir = 3
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
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_train.append([rri_interp_signal, ampl_interp_signal])
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1))
    y_train = np.array(y_train, dtype="float32")

    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_test.append([rri_interp_signal, ampl_interp_signal])
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    return (x_train, y_train, groups_train), (x_test, y_test, groups_test)

def build_custom_lenet5_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(6, (5, 5), activation='relu', padding='same')(inputs)
    x = AveragePooling2D(pool_size=(2, 1))(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 1))(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

    

if __name__ == "__main__":
    (x_train, y_train, groups_train), (x_test, y_test, groups_test) = load_data()

    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    model = build_custom_lenet5_model(input_shape)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    model.save(os.path.join(r"D:\finalyearproject\Sleep-apnea-detection-through-a-modified-LeNet-5-convolutional-neural-network-master\models", "custom_lenet5_model.h5"))

    # Evaluate the model
    print("training:") 
    y_true, y_pred = y_train, np.argmax(model.predict(x_train, batch_size=1024, verbose=1), axis=-1) 
    C = confusion_matrix(y_true, y_pred, labels=(1, 0)) 
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1] 
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP) 
    print("acc: {}, sn: {}, sp: {}".format(acc, sn, sp)) 


    print("testing:") 
    y_true, y_pred = y_test, np.argmax(model.predict(x_test, batch_size=1024, verbose=1), axis=-1) 
    C = confusion_matrix(y_true, y_pred, labels=(1, 0)) 
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1] 
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP) 
    print("acc: {}, sn: {}, sp: {}".format(acc, sn, sp))