import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from numpy.lib.stride_tricks import sliding_window_view

WINDOW_SIZE = 99

def load_csv_from_folder(folder, index=None, axis=0):
    script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in
    rel_path = f"{folder}/*.csv" #DataBases/
    abs_path = os.path.join(script_dir, rel_path)

    all_files = glob.glob(abs_path)
    all_files.sort()

    li = []

    for filename in all_files:
        if index is not None:
            df = pd.read_csv(filename, header=0, index_col=index, parse_dates=[index])
        else:
            df = pd.read_csv(filename, header=0)
        li.append(df)

    return pd.concat(li, axis=axis, ignore_index=False)


def create_model():
    model = Sequential()#add model layers
    model.add(Conv1D(30, kernel_size=10, activation="relu", strides=1, input_shape=(WINDOW_SIZE, 1)))
    model.add(Conv1D(30, kernel_size=8, activation="relu", strides=1))
    model.add(Conv1D(40, kernel_size=6, activation="relu", strides=1))
    #model.add(Dropout(0.1))
    model.add(Conv1D(50, kernel_size=5, activation="relu", strides=1))
    #model.add(Dropout(0.2))
    model.add(Conv1D(50, kernel_size=5, activation="relu", strides=1))
    #model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def create_dataset(dataset_X, dataset_Y, window_size=99):
    gap = int((window_size-1)/2)
    dataX, dataY = [], []
    # for i in range(len(dataset_X)-window_size-1):
    #     a = dataset_X.iloc[i:(i+window_size), 0]
    #     dataX.append(a)
    #     dataY.append(dataset_Y.iloc[i + int(window_size/2)])
    #     print(i)
    # dataX = np.reshape(np.array(dataX), [-1, window_size, 1])
    dataX = np.reshape(dataset_X.to_numpy(), [len(dataset_X)])
    dataX = sliding_window_view(dataX, window_size)
    index = dataset_Y.index[(int(window_size/2)):-(int(window_size/2))]
    dataY = np.array(dataset_Y.iloc[gap:-gap])
    dataY[dataY > 10] = 1
    return dataX, dataY, index