import os

from clearml import Dataset

from ClassificationUtils import create_binary_model, create_multiclassing_model, create_multilabeling_model
from DataUtils import get_all_devices


def init():
    model = create_binary_model()
    models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models"
    model.save_weights(models_dir + "/BinaryClassificationModel/model.h5")

    project_root = os.path.dirname(os.path.abspath(os.path.curdir))
    dataset = Dataset.create(dataset_project="Binary_Classification_Test", dataset_name="Models")
    dataset.add_files(path=project_root + '/Models')
    dataset.upload(chunk_size=100)
    dataset.finalize()

    dataset = Dataset.create(dataset_project="Binary_Classification_Test", dataset_name="DataBases")
    dataset.add_files(path=os.path.dirname(os.path.abspath(os.path.curdir)) + '/Resources')
    dataset.upload(chunk_size=100)
    dataset.finalize()

    dataset = Dataset.create(dataset_project="Binary_Classification_Test", dataset_name="Results")
    dataset.add_files(path=os.path.dirname(os.path.abspath(os.path.curdir)) + '/Results')
    dataset.upload(chunk_size=100)
    dataset.finalize()


def init_model(filepath: str):
    model = create_binary_model()
    model.save_weights(filepath)

    project_root = os.path.dirname(os.path.abspath(os.path.curdir))
    dataset = Dataset.create(dataset_project="Binary_Classification_Test", dataset_name="Models")
    dataset.add_files(path=project_root + '/Models')
    dataset.upload(chunk_size=100)
    dataset.finalize()


def init_binary_all_devices(folder: str):
    devices = get_all_devices(os.path.dirname(os.path.abspath(os.path.curdir))
                              + "/Resources/TimeDataWeeks/TimeSeriesData/Week0/2022-12-05.csv")

    for device in devices:
        model = create_binary_model()
        model.save_weights(folder + "/model_" + device + ".h5")

    project_root = os.path.dirname(os.path.abspath(os.path.curdir))
    dataset = Dataset.create(dataset_project='Binary_ClassificationAllDevices_Test', dataset_name="Models")
    dataset.add_files(path=project_root + '/Models/')
    dataset.upload(chunk_size=100)
    dataset.finalize()


def init_multiclassing(filepath: str):
    devices = get_all_devices(os.path.dirname(os.path.abspath(os.path.curdir))
                              + "/Resources/TimeDataWeeks/TimeSeriesData/Week0/2022-12-05.csv")
    model = create_multiclassing_model(len(devices))
    model.save_weights(filepath)

    project_root = os.path.dirname(os.path.abspath(os.path.curdir))
    dataset = Dataset.create(dataset_project="MultiClassing_Classification_Test", dataset_name="Models")
    dataset.add_files(path=project_root + '/Models')
    dataset.upload(chunk_size=100)
    dataset.finalize()


def init_multilabeling(filepath: str):
    devices = get_all_devices(os.path.dirname(os.path.abspath(os.path.curdir))
                              + "/Resources/TimeDataWeeks/TimeSeriesData/Week0/2022-12-05.csv")
    model = create_multilabeling_model(len(devices))
    model.save_weights(filepath)

    project_root = os.path.dirname(os.path.abspath(os.path.curdir))
    dataset = Dataset.create(dataset_project="MultiLabeling_Classification_Test", dataset_name="Models")
    dataset.add_files(path=project_root + '/Models')
    dataset.upload(chunk_size=100)
    dataset.finalize()


if __name__ == '__main__':
    # init()
    models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "/Models"
    # init_model(models_dir + "/BinaryClassificationModel/model.h5")
    # init_binary_all_devices(models_dir + "/BinaryClassificationAllDevices")
    init_multiclassing(models_dir + "/MultiClassingClassification/model.h5")
