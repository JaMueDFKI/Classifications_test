import clearml
from clearml import Dataset, Task
from keras import Model, Sequential
from tensorflow.python import keras

from BinaryClassificationUtils import load_csv_from_folder, create_dataset

from keras.losses import binary_crossentropy
from keras.metrics import Recall, Precision
from tensorflow_addons.metrics import F1Score
from keras.optimizers import Adam

RESAMPLING_RATE = "10s"


def start_task():
    task = Task.init(project_name='Binary_Classification_Test', task_name=f'Experiment Test Binary')
    task.execute_remotely(queue_name='default', clone=False, exit_process=True)

    # get local copy of DataBases
    dataset_databases = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='DataBases')
    dataset_path_databases = dataset_databases.get_mutable_local_copy("DataBases/", True)

    # get local copy of Results
    dataset_results = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='Results')
    dataset_path_results = dataset_results.get_mutable_local_copy("Results/", True)

    # get local copy of Results
    models = Dataset.get(dataset_project='Binary_Classification_Test', dataset_name='Models')
    models_path = models.get_mutable_local_copy("Models/", True)

    model = keras.models.load_model(models_path + "\\BinaryClassificationModel")

    dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week0"
    dataX = load_csv_from_folder(dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

    dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week0"
    dataY = load_csv_from_folder(dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()

    dataX, dataY, index = create_dataset(dataset_X=dataX.loc[:, "smartMeter"],
                                         dataset_Y=dataY.loc[:, "kettle"])

    val_dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week1"
    val_dataX = load_csv_from_folder(val_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

    val_dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week1"
    val_dataY = load_csv_from_folder(val_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()

    val_dataX, val_dataY, val_index = create_dataset(dataset_X=val_dataX.loc[:, "smartMeter"],
                                                     dataset_Y=val_dataY.loc[:, "kettle"])

    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=[F1Score(),
                           "accuracy",
                           Recall(name="recall"),
                           Precision(name="precision")])

    model.fit(x=dataX, y=dataY, epochs=10, validation_data=(val_dataX, val_dataY))

    test_dataX_folder = dataset_path_databases + "/TimeDataWeeks/TimeSeriesData/Week2"
    test_dataX = load_csv_from_folder(test_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()

    test_dataY_folder = dataset_path_databases + "/TimeDataWeeks/Active_phases/Week2"
    test_dataY = load_csv_from_folder(test_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()

    test_dataX, test_dataY, test_index = create_dataset(dataset_X=test_dataX.loc[:, "smartMeter"],
                                                        dataset_Y=test_dataY.loc[:, "kettle"])

    results = model.evaluate(test_dataX, test_dataY)

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

    dataset = Dataset.create(
        dataset_project="Binary_Classification_Test", dataset_name="DataBases"
    )
    dataset.add_files(path='DataBases/')
    dataset.upload(chunk_size=100)
    dataset.finalize()
    print("DataBases uploaded.")


if __name__ == '__main__':
    start_task()


