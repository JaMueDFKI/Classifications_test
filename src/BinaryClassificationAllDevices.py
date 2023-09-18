import os
from datetime import datetime

import clearml
from clearml import Dataset, Task
import tensorflow as tf
from keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Dense

from BinaryClassificationUtils import load_csv_from_folder, create_dataset, create_model, load_label_data

from tensorflow_addons.metrics import F1Score

from ClearMLProjectInit import init_binary_all_devices
from DataUtils import get_all_devices

RESAMPLING_RATE = "10s"


def start_task():
    task = Task.init(project_name='Binary_ClassificationAllDevices_Test',
                     task_name=f'Experiment Test Binary All Devices')
    task.execute_remotely(queue_name='default', clone=False, exit_process=True)

    # get local copy of DataBases
    dataset_databases = Dataset.get(dataset_project='Binary_ClassificationAllDevices_Test', dataset_name='DataBases')
    dataset_path_databases = dataset_databases.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/DataBases", True
    )

    # get local copy of Results
    dataset_results = Dataset.get(dataset_project='Binary_ClassificationAllDevices_Test', dataset_name='Results')
    dataset_path_results = dataset_results.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Results", True
    )

    # get local copy of Results
    models = Dataset.get(dataset_project='Binary_ClassificationAllDevices_Test', dataset_name='Models')
    models_path = models.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "Models/", True
    )

    devices = get_all_devices(dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week0/2022-12-05.csv")
    time_test_started = datetime.now().strftime("%Y%m%d-%H%M%S")

    fake_model_training()

    for device in devices:
        dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week0"
        dataX = load_csv_from_folder(dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

        dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week0"
        dataY = load_label_data(devices, dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()

        dataX, dataY, index = create_dataset(dataset_X=dataX.loc[:, "smartMeter"],
                                             dataset_Y=dataY.loc[:, device])

        val_dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week1"
        val_dataX = load_csv_from_folder(val_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

        val_dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week1"
        val_dataY = load_label_data(devices, val_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()

        val_dataX, val_dataY, val_index = create_dataset(dataset_X=val_dataX.loc[:, "smartMeter"],
                                                         dataset_Y=val_dataY.loc[:, device])
        model = create_model()

        model.compile(loss="binary_crossentropy", optimizer="adam",
                      metrics=["accuracy",
                               Recall(name="recall"),
                               Precision(name="precision")])

        model.load_weights(models_path + "/BinaryClassificationAllDevices/model_" + device + ".h5")

        logdir = (dataset_path_results + "/BinaryClassificationAllDevices/training/" + time_test_started)

        if not os.path.exists(logdir):
            os.mkdir(logdir)

        tensorboard_callback = TensorBoard(log_dir=logdir + "/" + device)
        csv_callback = CSVLogger(logdir + "/results_" + device + ".csv")

        print("Start training " + device)

        training_results = model.fit(x=dataX, y=dataY, epochs=10, validation_data=(val_dataX, val_dataY),
                                     callbacks=[tensorboard_callback, csv_callback])
        print(training_results)

        test_dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week2"
        test_dataX = load_csv_from_folder(test_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

        test_dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week2"
        test_dataY = load_label_data(devices, test_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()

        test_dataX, test_dataY, test_index = create_dataset(dataset_X=test_dataX.loc[:, "smartMeter"],
                                                            dataset_Y=test_dataY.loc[:, device])

        logdir = (dataset_path_results + "/BinaryClassificationAllDevices/test/" + time_test_started)

        if not os.path.exists(logdir):
            os.mkdir(logdir)

        tensorboard_callback = TensorBoard(log_dir=logdir + "/" + device)
        csv_callback = CSVLogger(logdir + "/results_" + device + ".csv")

        csv_callback.on_test_begin = csv_callback.on_train_begin
        csv_callback.on_test_batch_end = csv_callback.on_epoch_end
        csv_callback.on_test_end = csv_callback.on_train_end

        print("Start Evaluation " + device)

        model.evaluate(test_dataX, test_dataY, callbacks=[tensorboard_callback, csv_callback])

        models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models"
        model.save_weights(models_dir + "/BinaryClassificationAllDevices/model_" + device + ".h5", overwrite=True)

    project_root = os.path.dirname(os.path.abspath(os.path.curdir))

    # save the Results of the Model for experiment_number
    dataset = Dataset.create(
        dataset_project='Binary_ClassificationAllDevices_Test', dataset_name="Results"
    )
    dataset.add_files(path=project_root + '/Results')
    dataset.upload(chunk_size=100)
    dataset.finalize()
    print("Results uploaded.")

    # save the Model for experiment_number
    dataset = Dataset.create(
        dataset_project='Binary_ClassificationAllDevices_Test', dataset_name="Models"
    )
    dataset.add_files(path=project_root + '/Models')
    dataset.upload(chunk_size=100)
    dataset.finalize()
    print("Models uploaded.")


def init_test():
    models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models"
    init_binary_all_devices(models_dir + "/BinaryClassificationAllDevices")

    dataset = Dataset.create(dataset_project='Binary_ClassificationAllDevices_Test', dataset_name="DataBases")
    dataset.add_files(path=os.path.dirname(os.path.abspath(os.path.curdir)) + '/Resources')
    dataset.upload(chunk_size=100)
    dataset.finalize()

    dataset = Dataset.create(dataset_project='Binary_ClassificationAllDevices_Test', dataset_name="Results")
    dataset.add_files(path=os.path.dirname(os.path.abspath(os.path.curdir)) + '/Results')
    dataset.upload(chunk_size=100)
    dataset.finalize()


def get_datasets_from_remote():
    # get local copy of DataBases
    dataset_databases = Dataset.get(dataset_project='Binary_ClassificationAllDevices_Test', dataset_name='DataBases')
    dataset_path_databases = dataset_databases.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/DataBases", True
    )

    # get local copy of Results
    dataset_results = Dataset.get(dataset_project='Binary_ClassificationAllDevices_Test', dataset_name='Results')
    dataset_path_results = dataset_results.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Results", True
    )

    # get local copy of Models
    models = Dataset.get(dataset_project='Binary_ClassificationAllDevices_Test', dataset_name='Models')
    models_path = models.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models", True
    )


def create_fake_model():
    model = Sequential()
    model.add(Dense(1, activation='sigmoid'))
    return model


def fake_model_training():
    """
    Trains a fake model for one epoch to display the graphs for the devices correctly in CleraML.
    """
    fake_model = create_fake_model()

    fake_model.compile(loss="binary_crossentropy", optimizer="adam",
                       metrics=["accuracy",
                                Recall(name="recall"),
                                Precision(name="precision")])

    tensorboard_callback = TensorBoard()

    fake_model.fit(x=[1], y=[1], epochs=1, validation_data=([1], [1]),
                   callbacks=[tensorboard_callback])

    fake_model.evaluate(x=[1], y=[1], callbacks=[tensorboard_callback])


if __name__ == '__main__':
    # init_test()
    start_task()
    # get_datasets_from_remote()



