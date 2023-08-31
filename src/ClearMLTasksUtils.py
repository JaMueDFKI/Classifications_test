import clearml
from clearml import Dataset, Task


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
