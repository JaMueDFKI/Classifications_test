import os
from datetime import datetime

import clearml
import numpy as np
import pandas as pd
from clearml import Dataset, Task
import tensorflow as tf
from keras.optimizers import Adam
from keras.src.callbacks import CSVLogger
from keras.src.metrics import Recall, Precision, F1Score
from keras.src.optimizers.schedules import ExponentialDecay, PolynomialDecay
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard, EarlyStopping

from ClassificationUtils import load_csv_from_folder, create_dataset, load_label_data, create_multilabeling_model, \
    WeightedF1Score, F1ScoreClass

from ClearMLProjectModelInit import init_multilabeling
from DataUtils import get_all_devices_file, get_all_devices_data

RESAMPLING_RATE = "4s"


def start_task():
    task = Task.init(project_name='MultiLabeling_Classification_Test',
                     task_name=f'Experiment Test MultiLabeling ('
                               f'resampling rate= 4s,'
                               f' activation_function=leaky_relu,'
                               # f' learning_rate=0.00001,'
                               # f'exponential lr scheduler(0.001, 61422, 0.9, True)'
                               f'polynomial lr scheduler(0.001, 61422*40, 0.00001, 2, False)'
                               # f' w\\ Dropout'
                               # f' additional layer'
                               f')')
    task.execute_remotely(queue_name='default', clone=False, exit_process=True)

    # get local copy of DataBases
    dataset_databases = Dataset.get(dataset_project='MultiLabeling_Classification_Test', dataset_name='DataBases')
    dataset_path_databases = dataset_databases.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/DataBases", True
    )

    # get local copy of Results
    dataset_results = Dataset.get(dataset_project='MultiLabeling_Classification_Test', dataset_name='Results')
    dataset_path_results = dataset_results.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Results", True
    )

    # get local copy of Results
    models = Dataset.get(dataset_project='MultiLabeling_Classification_Test', dataset_name='Models')
    models_path = models.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models", True
    )

    devices = get_all_devices_data(dataset_path_databases + "/TimeDataWeeksOnlyUsedDevices/TimeSeriesData")
    devices.remove('vacuum cleaner')
    time_test_started = datetime.now().strftime("%Y%m%d-%H%M%S")

    week_counter = 0
    data_X_folders = []
    data_Y_folders = []

    while week_counter < 13:
        data_X_folders.append(
            dataset_path_databases + "/TimeDataWeeksOnlyUsedDevices/TimeSeriesData/Week" + str(week_counter))
        data_Y_folders.append(
            dataset_path_databases + "/TimeDataWeeksOnlyUsedDevices/Active_phases/Week" + str(week_counter))
        week_counter += 1

    dataX, dataY, index = get_multilabeling_dataset(data_X_folders, data_Y_folders, devices)

    val_data_X_folders = []
    val_data_Y_folders = []

    while week_counter < 26:
        val_data_X_folders.append(
            dataset_path_databases + "/TimeDataWeeksOnlyUsedDevices/TimeSeriesData/Week" + str(week_counter))
        val_data_Y_folders.append(
            dataset_path_databases + "/TimeDataWeeksOnlyUsedDevices/Active_phases/Week" + str(week_counter))
        week_counter += 1

    val_dataX, val_dataY, val_index = get_multilabeling_dataset(val_data_X_folders, val_data_Y_folders, devices)

    model = create_multilabeling_model(len(devices))

    device_pointer = 0
    metrics = []
    # f1_weigths = []
    for device in devices:
        metrics.append(Recall(name="recall_" + device, class_id=device_pointer))
        metrics.append(Precision(name="precision_" + device, class_id=device_pointer))
        metrics.append(F1ScoreClass(name="f1_score_" + device, class_id=device_pointer))

        # match device:
        #     case "kettle" | "coffee machine": f1_weigths.append(0.2)
        #     case "computer" | "microwave": f1_weigths.append(0.3)
        #     case "vacuum cleaner": f1_weigths.append(0)
        device_pointer += 1

    # f1_weigths = np.array(f1_weigths)
    # print(f1_weigths)
    # print(devices)

    # metrics.append(WeightedF1Score(name="weighted_f1_score", num_classes=len(devices), weights=f1_weigths))

    # metrics.append(F1Score())
    metrics.append(F1Score(average="macro", name="averaged_f1_score"))

    learning_rate = PolynomialDecay(initial_learning_rate=0.001,
                                    decay_steps=61422*40,
                                    end_learning_rate=0.00001,
                                    power=2,
                                    cycle=False)

    # learning_rate = ExponentialDecay(initial_learning_rate=0.001,
    #                                  decay_steps=61422,
    #                                  decay_rate=0.9,
    #                                  staircase=True)

    adam_opt = Adam(learning_rate=learning_rate)

    model.compile(loss="binary_crossentropy", optimizer=adam_opt, metrics=metrics)

    model.load_weights(models_path + "/MultiLabelingClassificationModel/model.h5")

    logdir = (dataset_path_results + "/MultiLabelingClassification/training/" + time_test_started)

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    tensorboard_callback = TensorBoard(log_dir=logdir)
    csv_callback = CSVLogger(logdir + "/results.csv")
    early_stopping_callback = EarlyStopping(monitor='val_averaged_f1_score',
                                            patience=10,
                                            mode='max',
                                            start_from_epoch=10)

    print("Start training ")

    training_results = model.fit(x=dataX, y=dataY, epochs=50, validation_data=(val_dataX, val_dataY),
                                 callbacks=[tensorboard_callback, csv_callback, early_stopping_callback])
    print(training_results)

    # test_dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week2"
    # test_dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week2"

    # test_data_X_folders = []
    # test_data_Y_folders = []

    # while week_counter < 26:
    #     test_data_X_folders.append(
    #         dataset_path_databases + "/TimeDataWeeksOnlyUsedDevices/TimeSeriesData/Week" + str(week_counter))
    #     test_data_Y_folders.append(
    #         dataset_path_databases + "/TimeDataWeeksOnlyUsedDevices/Active_phases/Week" + str(week_counter))
    #     week_counter += 1

    # test_dataX, test_dataY, test_index = get_multilabeling_dataset(test_data_X_folders, test_data_Y_folders, devices)

    # logdir = (dataset_path_results + "/MultiLabelingClassification/test/" + time_test_started)

    # if not os.path.exists(logdir):
    #     os.mkdir(logdir)

    # tensorboard_callback = TensorBoard(log_dir=logdir)
    # csv_callback = CSVLogger(logdir + "/results.csv")

    # csv_callback.on_test_begin = csv_callback.on_train_begin
    # csv_callback.on_test_batch_end = csv_callback.on_epoch_end
    # csv_callback.on_test_end = csv_callback.on_train_end

    # print("Start Evaluation ")

    # model.evaluate(test_dataX, test_dataY, callbacks=[tensorboard_callback, csv_callback])

    models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models"
    model.save_weights(models_dir + "/MultiLabelingClassificationModel/model.h5", overwrite=True)

    project_root = os.path.dirname(os.path.abspath(os.path.curdir))

    # save the Results of the Model for experiment_number
    dataset = Dataset.create(
        dataset_project='MultiLabeling_Classification_Test', dataset_name="Results"
    )
    dataset.add_files(path=project_root + '/Results')
    dataset.upload(chunk_size=100)
    dataset.finalize()
    print("Results uploaded.")

    # save the Model for experiment_number
    dataset = Dataset.create(
        dataset_project='MultiLabeling_Classification_Test', dataset_name="Models"
    )
    dataset.add_files(path=project_root + '/Models')
    dataset.upload(chunk_size=100)
    dataset.finalize()
    print("Models uploaded.")


def init_test():
    models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models"
    init_multilabeling(models_dir + "/MultiLabelingClassificationModel/model.h5",
                       os.path.dirname(os.path.abspath(os.path.curdir))
                       + '/Resources/TimeDataWeeksOnlyUsedDevices/TimeSeriesData')

    dataset = Dataset.create(dataset_project='MultiLabeling_Classification_Test', dataset_name="DataBases")
    dataset.add_files(path=os.path.dirname(os.path.abspath(os.path.curdir)) + '/Resources')
    dataset.upload(chunk_size=100)
    dataset.finalize()

    dataset = Dataset.create(dataset_project='MultiLabeling_Classification_Test', dataset_name="Results")
    dataset.add_files(path=os.path.dirname(os.path.abspath(os.path.curdir)) + '/Results')
    dataset.upload(chunk_size=100)
    dataset.finalize()


def get_datasets_from_remote():
    # get local copy of DataBases
    dataset_databases = Dataset.get(dataset_project='MultiLabeling_Classification_Test', dataset_name='DataBases')
    dataset_path_databases = dataset_databases.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/DataBases", True
    )

    # get local copy of Results
    dataset_results = Dataset.get(dataset_project='MultiLabeling_Classification_Test', dataset_name='Results')
    dataset_path_results = dataset_results.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Results", True
    )

    # get local copy of Models
    models = Dataset.get(dataset_project='MultiLabeling_Classification_Test', dataset_name='Models')
    models_path = models.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models", True
    )


def get_multilabeling_dataset(data_x_folders: list[str], data_y_folders: list[str], devices: list[str]):
    data_X = []

    for data_x_folder in data_x_folders:
        data_X.append(load_csv_from_folder(data_x_folder, index="timestamp").resample(RESAMPLING_RATE).mean())

    dataX = pd.concat(data_X, axis=0, ignore_index=False)

    min_max_scaler = MinMaxScaler()
    dataX_scaled = pd.DataFrame(
        min_max_scaler.fit_transform(dataX))
    dataX_scaled.index = dataX.index
    dataX_scaled.columns = dataX.columns

    data_Y = []

    for data_y_folder in data_y_folders:
        data_Y.append(
            load_label_data(devices, data_y_folder, index="timestamp").reindex(devices, axis=1)
            .resample(RESAMPLING_RATE).median()
        )

    dataY = pd.concat(data_Y, axis=0, ignore_index=False)

    return create_dataset(dataset_X=dataX_scaled.loc[:, "smartMeter"], dataset_Y=dataY)


if __name__ == '__main__':
    # init_test()
    start_task()
    # get_datasets_from_remote()
