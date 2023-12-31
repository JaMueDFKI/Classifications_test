import os
from datetime import datetime

import clearml
from clearml import Dataset, Task
import tensorflow as tf
from keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.python.keras.callbacks import CSVLogger

from ClassificationUtils import load_csv_from_folder, create_dataset, create_binary_model, load_label_data

from tensorflow_addons.metrics import F1Score

from DataUtils import get_all_devices_file

RESAMPLING_RATE = "10s"


def start_task():
    task = Task.init(project_name='Binary_Classification_Test',
                     task_name=f'Experiment Test Binary ')
    task.execute_remotely(queue_name='default', clone=False, exit_process=True)

    # get local copy of DataBases
    dataset_databases = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='DataBases')
    dataset_path_databases = dataset_databases.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/DataBases", True
    )

    # get local copy of Results
    dataset_results = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='Results')
    dataset_path_results = dataset_results.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Results", True
    )

    # get local copy of Models
    models = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='Models')
    models_path = models.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models", True
    )

    dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week0"
    dataX = load_csv_from_folder(dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

    devices = get_all_devices_file(dataX_folder + "/2022-12-05.csv")

    dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week0"
    dataY = load_label_data(devices, dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()

    dataX, dataY, index = create_dataset(dataset_X=dataX.loc[:, "smartMeter"],
                                         dataset_Y=dataY.loc[:, "kettle"])

    val_dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week1"
    val_dataX = load_csv_from_folder(val_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

    val_dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week1"
    val_dataY = load_label_data(devices, val_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()

    val_dataX, val_dataY, val_index = create_dataset(dataset_X=val_dataX.loc[:, "smartMeter"],
                                                     dataset_Y=val_dataY.loc[:, "kettle"])
    model = create_binary_model()

    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=[  # F1Score(num_classes=1, dtype=float),
                      "accuracy",
                      Recall(name="recall"),
                      Precision(name="precision")])

    model.load_weights(models_path + "/BinaryClassificationModel/model.h5")

    time_test_started = datetime.now().strftime("%Y%m%d-%H%M%S")

    logdir = (dataset_path_results + "/BinaryClassification/training/" + time_test_started)

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    tensorboard_callback = TensorBoard(log_dir=logdir)
    csv_callback = CSVLogger(logdir + "/results_csv.csv")

    print("Start training")
    model.fit(x=dataX, y=dataY, epochs=10, validation_data=(val_dataX, val_dataY),
              callbacks=[tensorboard_callback, csv_callback])

    test_dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week2"
    test_dataX = load_csv_from_folder(test_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

    test_dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week2"
    test_dataY = load_label_data(devices, test_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()

    test_dataX, test_dataY, test_index = create_dataset(dataset_X=test_dataX.loc[:, "smartMeter"],
                                                        dataset_Y=test_dataY.loc[:, "kettle"])

    logdir = (dataset_path_results + "/BinaryClassification/test/" + time_test_started)

    tensorboard_callback = TensorBoard(log_dir=logdir)
    csv_callback = CSVLogger(logdir + "/results_csv.csv")

    print("test started")
    model.evaluate(test_dataX, test_dataY, callbacks=[tensorboard_callback, csv_callback])

    models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models"
    model.save_weights(models_dir + "/BinaryClassificationModel/model.h5", overwrite=True)

    project_root = os.path.dirname(os.path.abspath(os.path.curdir))

    # save the Results of the Model for experiment_number
    dataset = Dataset.create(
        dataset_project="Binary_Classification_Test", dataset_name="Results"
    )
    dataset.add_files(path=project_root + '/Results')
    dataset.upload(chunk_size=100)
    dataset.finalize()
    print("Results uploaded.")

    # save the Model for experiment_number
    dataset = Dataset.create(
        dataset_project="Binary_Classification_Test", dataset_name="Models"
    )
    dataset.add_files(path=project_root + '/Models')
    dataset.upload(chunk_size=100)
    dataset.finalize()
    print("Models uploaded.")

    # dataset = Dataset.create(
    #     dataset_project="Binary_Classification_Test", dataset_name="DataBases"
    # )
    # dataset.add_files(path='DataBases/')
    # dataset.upload(chunk_size=100)
    # dataset.finalize()
    # print("DataBases uploaded.")


def get_datasets_from_remote():
    # get local copy of DataBases
    dataset_databases = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='DataBases')
    dataset_path_databases = dataset_databases.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/DataBases", True
    )

    # get local copy of Results
    dataset_results = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='Results')
    dataset_path_results = dataset_results.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Results", True
    )

    # get local copy of Models
    models = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='Models')
    models_path = models.get_mutable_local_copy(
        os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models", True
    )


if __name__ == '__main__':
    start_task()
    # get_datasets_from_remote()
