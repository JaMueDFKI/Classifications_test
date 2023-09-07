import os

from clearml import Dataset

from BinaryClassificationUtils import create_model


def init():
    model = create_model()
    models_dir = os.path.dirname(os.path.abspath(os.path.curdir)) + "\\Models"
    model.save_weights(models_dir + "\\BinaryClassificationModel/model.h5")

    project_root = os.path.dirname(os.path.abspath(os.path.curdir))
    dataset = Dataset.create(dataset_project="Binary_Classification_Test", dataset_name="Models")
    dataset.add_files(path=project_root + '\\Models/')
    dataset.upload(chunk_size=100)
    dataset.finalize()

    dataset = Dataset.create(dataset_project="Binary_Classification_Test", dataset_name="DataBases")
    dataset.add_files(path=os.path.dirname(os.path.abspath(os.path.curdir)) + '\\Resources/')
    dataset.upload(chunk_size=100)
    dataset.finalize()

    dataset = Dataset.create(dataset_project="Binary_Classification_Test", dataset_name="Results")
    dataset.upload(chunk_size=100)
    dataset.finalize()


if __name__ == '__main__':
    init()
