import os
from datetime import datetime

import clearml
from clearml import Dataset, Task
import tensorflow as tf
from keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Recall, Precision

from BinaryClassificationUtils import load_csv_from_folder, create_dataset, create_model, load_label_data

from tensorflow_addons.metrics import F1Score

from DataUtils import get_all_devices

RESAMPLING_RATE = "10s"


def start_task():
    # get local copy of DataBases
    dataset_databases = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='DataBases')
    dataset_path_databases = dataset_databases.get_mutable_local_copy("DataBases/", True)

    # get local copy of Results
    dataset_results = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='Results')
    dataset_path_results = dataset_results.get_mutable_local_copy("Results/", True)

    # get local copy of Results
    models = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='Models')
    models_path = models.get_mutable_local_copy("Models/", True)

    devices = get_all_devices(dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week0/2022-12-05.csv")

    for device in devices:

        task = Task.init(project_name='Binary_Classification_Test',
                         task_name=f'Experiment Test Binary ' + device)
        task.execute_remotely(queue_name='default', clone=False, exit_process=True)


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

        logdir = (os.path.dirname(os.path.abspath(os.path.curdir)) + "logs/scalars/training/"
                  + datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=logdir)

        training_results = model.fit(x=dataX, y=dataY, epochs=10, validation_data=(val_dataX, val_dataY),
                                     callbacks=[tensorboard_callback])
        print(training_results)

        test_dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week2"
        test_dataX = load_csv_from_folder(test_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

        test_dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week2"
        test_dataY = load_label_data(devices, test_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()

        test_dataX, test_dataY, test_index = create_dataset(dataset_X=test_dataX.loc[:, "smartMeter"],
                                                            dataset_Y=test_dataY.loc[:, device])

        results = model.evaluate(test_dataX, test_dataY, callbacks=[tensorboard_callback])

        models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models"
        model.save_weights(models_dir + "/BinaryClassificationAllDevices/model_" + device + ".h5", overwrite=True)

        # save the Results of the Model for experiment_number
        dataset = Dataset.create(
            dataset_project="Binary_Classification_Test", dataset_name="Results"
        )
        dataset.add_files(path='Results/')
        dataset.upload(chunk_size=100)
        dataset.finalize()
        print("Results uploaded.")

        # save the Model for experiment_number
        dataset = Dataset.create(
            dataset_project="Binary_Classification_Test", dataset_name="Models"
        )
        dataset.add_files(path='Models/')
        dataset.upload(chunk_size=100)
        dataset.finalize()
        print("Models uploaded.")


if __name__ == '__main__':
    start_task()


