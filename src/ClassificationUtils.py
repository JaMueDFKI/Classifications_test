import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.metrics.base_metric
import keras.backend as K
from keras.layers import Dropout
from keras.metrics import Recall, Precision
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from numpy.lib.stride_tricks import sliding_window_view

WINDOW_SIZE = 99


def load_csv_from_folder(folder, index=None, axis=0) -> pd.DataFrame:
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


def load_label_data(devices, folder, index=None, axis=0) -> pd.DataFrame:
    csv_data = load_csv_from_folder(folder, index, axis)
    csv_data.fillna(0, inplace=True)
    csv_data = csv_data.astype(int)

    csv_columns = csv_data.columns
    for device in devices:
        if device not in csv_columns:
            csv_data[device] = [0] * len(csv_data.index)

    return csv_data


def add_idle(label_data: pd.DataFrame) -> pd.DataFrame:
    sum_ser = label_data.sum(axis=1)
    idle_vector = sum_ser.apply(lambda x: 0 if x > 0 else 1)
    label_data["Idle"] = idle_vector
    return label_data


def create_binary_model():
    model = Sequential()#add model layers
    model.add(Conv1D(30, kernel_size=10, activation="leaky_relu", strides=1, input_shape=(WINDOW_SIZE, 1)))
    model.add(Conv1D(30, kernel_size=8, activation="leaky_relu", strides=1))
    model.add(Conv1D(40, kernel_size=6, activation="leaky_relu", strides=1))
    # model.add(Dropout(0.1))
    model.add(Conv1D(50, kernel_size=5, activation="leaky_relu", strides=1))
    # model.add(Dropout(0.2))
    model.add(Conv1D(50, kernel_size=5, activation="leaky_relu", strides=1))
    # model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1024, activation='leaky_relu'))
    # model.add(Dropout(0.5))
    # additional fully connected layer
    model.add(Dense(1024, activation='leaky_relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def create_multiclassing_model(number_devices: int):
    """
    Creates a multiclassing model.
    :param number_devices: The number of devices, for which the multiclassing model should be created.
    :return: A multiclassing model for number_devices classes.
    """
    model = Sequential()  # add model layers
    model.add(Conv1D(30, kernel_size=10, activation="relu", strides=1, input_shape=(WINDOW_SIZE, 1)))
    model.add(Conv1D(30, kernel_size=8, activation="relu", strides=1))
    model.add(Conv1D(40, kernel_size=6, activation="relu", strides=1))
    # model.add(Dropout(0.1))
    model.add(Conv1D(50, kernel_size=5, activation="relu", strides=1))
    # model.add(Dropout(0.2))
    model.add(Conv1D(50, kernel_size=5, activation="relu", strides=1))
    # model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    # number_of_devices + idle node
    model.add(Dense(number_devices + 1, activation='softmax'))
    return model


def create_multilabeling_model(number_devices: int):
    """
    Creates a multilabeling model.
    :param number_devices: The number of devices, for which the multilabeling model should be created.
    :return: A multilabeling model for number_devices classes.
    """
    model = Sequential()  # add model layers
    model.add(Conv1D(30, kernel_size=10, activation='leaky_relu', strides=1, input_shape=(WINDOW_SIZE, 1)))
    model.add(Conv1D(30, kernel_size=8, activation='leaky_relu', strides=1))
    model.add(Conv1D(40, kernel_size=6, activation='leaky_relu', strides=1))
    # model.add(Dropout(0.1))
    model.add(Conv1D(50, kernel_size=5, activation='leaky_relu', strides=1))
    # model.add(Dropout(0.2))
    model.add(Conv1D(50, kernel_size=5, activation='leaky_relu', strides=1))
    # model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1024, activation='leaky_relu'))
    # model.add(Dropout(0.5))
    # additional fully connected layer
    # model.add(Dense(1024, activation='leaky_relu'))
    # model.add(Dropout(0.5))
    # number_of_devices
    model.add(Dense(number_devices, activation='sigmoid'))
    return model


def create_dataset(dataset_X, dataset_Y, window_size=WINDOW_SIZE):
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


def f1_score_binary(y_true, y_pred) -> float:

    def recall_m(y_true, y_pred) -> float:
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall

    def precision_m(y_true, y_pred) -> float:
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class F1Score(keras.metrics.base_metric.Metric):

    def __init__(self, name, class_id: int = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.recall = Recall(class_id=class_id, **kwargs)
        self.precision = Precision(class_id=class_id, **kwargs)
        self.f1_score = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.recall.update_state(y_true, y_pred, sample_weight)
        self.precision.update_state(y_true, y_pred, sample_weight)
        recall = self.recall.result()
        precision = self.precision.result()
        self.f1_score = 2*((precision*recall)/(precision+recall+K.epsilon()))

    def result(self) -> float:
        return self.f1_score


class WeightedF1Score(keras.metrics.base_metric.Metric):

    def __init__(self, name: str, num_classes: int, weights: np.ndarray, **kwargs):
        super().__init__(name=name, **kwargs)
        if num_classes != len(weights):
            raise ValueError("The number of classes must be equal to the number of class_weights")
        self.class_weights = weights
        self.f1_scores = [F1Score(name=name, class_id=i, **kwargs) for i in range(num_classes)]
        self.weighted_f1_score = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        result = 0
        for i in range(len(self.class_weights)):
            self.f1_scores[i].update_state(y_true, y_pred, sample_weight)
            result += self.class_weights[i] * self.f1_scores[i].result()
        self.weighted_f1_score = result

    def result(self) -> float:
        return self.weighted_f1_score


if __name__ == '__main__':
    dataY_folder = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Resources/TimeDataWeeks/Active_phases/Week0"
    devices = ["kettle", "computer", "coffee machine", "microwave", "television"]
    data_Y = load_label_data(devices, dataY_folder, index="timestamp")
    label_with_idle = add_idle(data_Y)

    print(data_Y)