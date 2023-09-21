import os
from datetime import datetime

import clearml
import pandas as pd
from clearml import Dataset, Task
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.integration_test.preprocessing_test_utils import preprocessing
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.python.keras.callbacks import CSVLogger

from BinaryClassificationUtils import load_csv_from_folder, create_dataset, load_label_data, \
    create_multiclassing_model, add_idle

from ClearMLProjectModelInit import init_multiclassing
from DataUtils import get_all_devices

RESAMPLING_RATE = "10s"


def start_task():
    task = Task.init(project_name='MultiClassing_Classification_Test',
                     task_name=f'Experiment Test Multiclassing')
    task.execute_remotely(queue_name='default', clone=False, exit_process=True)

    # get local copy of DataBases
    dataset_databases = Dataset.get(dataset_project='MultiClassing_Classification_Test', dataset_name='DataBases')
    dataset_path_databases = dataset_databases.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/DataBases", True
    )

    # get local copy of Results
    dataset_results = Dataset.get(dataset_project='MultiClassing_Classification_Test', dataset_name='Results')
    dataset_path_results = dataset_results.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Results", True
    )

    # get local copy of Results
    models = Dataset.get(dataset_project='MultiClassing_Classification_Test', dataset_name='Models')
    models_path = models.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models", True
    )

    devices = get_all_devices(dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week0/2022-12-05.csv")
    time_test_started = datetime.now().strftime("%Y%m%d-%H%M%S")


    dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week0"
    dataX = load_csv_from_folder(dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

    min_max_scaler = preprocessing.MinMaxScaler()
    dataX = pd.DataFrame(
        min_max_scaler.fit_transform(dataX.values.reshape([-1, 1])))
    dataX.index = dataX.index

    dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week0"
    dataY = add_idle(
        load_label_data(devices, dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()
    )

    dataX, dataY, index = create_dataset(dataset_X=dataX.loc[:, "smartMeter"],
                                         dataset_Y=dataY)

    val_dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week1"
    val_dataX = load_csv_from_folder(val_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

    val_dataX = pd.DataFrame(
        min_max_scaler.fit_transform(val_dataX.values.reshape([-1, 1])))
    val_dataX.index = val_dataX.index

    val_dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week1"
    val_dataY = add_idle(
        load_label_data(devices, val_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()
    )

    val_dataX, val_dataY, val_index = create_dataset(dataset_X=val_dataX.loc[:, "smartMeter"],
                                                     dataset_Y=val_dataY)
    model = create_multiclassing_model(len(devices))

    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy",
                           Recall(name="recall"),
                           Precision(name="precision")])

    model.load_weights(models_path + "/MultiClassingClassificationModel/model.h5")

    logdir = (dataset_path_results + "/MultiClassingClassification/training/" + time_test_started)

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    tensorboard_callback = TensorBoard(log_dir=logdir)
    csv_callback = CSVLogger(logdir + "/results.csv")

    print("Start training ")

    training_results = model.fit(x=dataX, y=dataY, epochs=10, validation_data=(val_dataX, val_dataY),
                                 callbacks=[tensorboard_callback, csv_callback])
    print(training_results)

    test_dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week2"
    test_dataX = load_csv_from_folder(test_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

    test_dataX = pd.DataFrame(
        min_max_scaler.fit_transform(test_dataX.values.reshape([-1, 1])))
    test_dataX.index = test_dataX.index

    test_dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week2"
    test_dataY = add_idle(
        load_label_data(devices, test_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()
    )

    test_dataX, test_dataY, test_index = create_dataset(dataset_X=test_dataX.loc[:, "smartMeter"],
                                                        dataset_Y=test_dataY)

    logdir = (dataset_path_results + "/MultiClassingClassification/test/" + time_test_started)

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    tensorboard_callback = TensorBoard(log_dir=logdir)
    csv_callback = CSVLogger(logdir + "/results.csv")

    csv_callback.on_test_begin = csv_callback.on_train_begin
    csv_callback.on_test_batch_end = csv_callback.on_epoch_end
    csv_callback.on_test_end = csv_callback.on_train_end

    print("Start Evaluation ")

    model.evaluate(test_dataX, test_dataY, callbacks=[tensorboard_callback, csv_callback])

    models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models"
    model.save_weights(models_dir + "/MultiClassingClassificationModel/model.h5", overwrite=True)

    project_root = os.path.dirname(os.path.abspath(os.path.curdir))

    # save the Results of the Model for experiment_number
    dataset = Dataset.create(
        dataset_project='MultiClassing_Classification_Test', dataset_name="Results"
    )
    dataset.add_files(path=project_root + '/Results')
    dataset.upload(chunk_size=100)
    dataset.finalize()
    print("Results uploaded.")

    # save the Model for experiment_number
    dataset = Dataset.create(
        dataset_project='MultiClassing_Classification_Test', dataset_name="Models"
    )
    dataset.add_files(path=project_root + '/Models')
    dataset.upload(chunk_size=100)
    dataset.finalize()
    print("Models uploaded.")


def init_test():
    models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models"
    init_multiclassing(models_dir + "/MultiClassingClassificationModel/model.h5")

    dataset = Dataset.create(dataset_project='MultiClassing_Classification_Test', dataset_name="DataBases")
    dataset.add_files(path=os.path.dirname(os.path.abspath(os.path.curdir)) + '/Resources')
    dataset.upload(chunk_size=100)
    dataset.finalize()

    dataset = Dataset.create(dataset_project='MultiClassing_Classification_Test', dataset_name="Results")
    dataset.add_files(path=os.path.dirname(os.path.abspath(os.path.curdir)) + '/Results')
    dataset.upload(chunk_size=100)
    dataset.finalize()


def get_datasets_from_remote():
    # get local copy of DataBases
    dataset_databases = Dataset.get(dataset_project='MultiClassing_Classification_Test', dataset_name='DataBases')
    dataset_path_databases = dataset_databases.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/DataBases", True
    )

    # get local copy of Results
    dataset_results = Dataset.get(dataset_project='MultiClassing_Classification_Test', dataset_name='Results')
    dataset_path_results = dataset_results.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Results", True
    )

    # get local copy of Models
    models = Dataset.get(dataset_project='MultiClassing_Classification_Test', dataset_name='Models')
    models_path = models.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models", True
    )


if __name__ == '__main__':
    # init_test()
    start_task()
    # get_datasets_from_remote()
