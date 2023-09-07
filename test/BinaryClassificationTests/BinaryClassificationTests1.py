import os.path
import unittest

from keras.losses import binary_crossentropy
from keras.metrics import Recall, Precision
from tensorflow_addons.metrics import F1Score
from keras.optimizers import Adam

from BinaryClassificationUtils import load_csv_from_folder, create_dataset, create_model
RESAMPLING_RATE = "10s"


class BinaryClassificationTests(unittest.TestCase):
    def test_binary_classification_1(self):
        dataX_folder = (os.path.dirname(os.path.abspath(os.path.curdir))
                        + "/Resources/TimeDataWeeks/TimeSeriesData/Week0")
        dataX = load_csv_from_folder(dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()
        print(dataX)

        dataY_folder = (os.path.dirname(os.path.abspath(os.path.curdir))
                        + "/Resources/TimeDataWeeks/Active_phases/Week0")
        dataY = load_csv_from_folder(dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()
        print(dataY)

        dataX, dataY, index = create_dataset(dataset_X=dataX.loc[:, "smartMeter"],
                                             dataset_Y=dataY.loc[:, "kettle"])
        print(dataX)
        print(dataY)
        print(index)

        val_dataX_folder = (os.path.dirname(os.path.abspath(os.path.curdir))
                            + "/Resources/TimeDataWeeks/TimeSeriesData/Week1")
        val_dataX = load_csv_from_folder(val_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()
        print(val_dataX)

        val_dataY_folder = (os.path.dirname(os.path.abspath(os.path.curdir))
                            + "/Resources/TimeDataWeeks/Active_phases/Week1")
        val_dataY = load_csv_from_folder(val_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()
        print(val_dataY)

        val_dataX, val_dataY, val_index = create_dataset(dataset_X=val_dataX.loc[:, "smartMeter"],
                                                         dataset_Y=val_dataY.loc[:, "kettle"])
        print(val_dataX)
        print(val_dataY)
        print(val_index)

        model = create_model()
        model.compile(loss="binary_crossentropy", optimizer="adam",
                      metrics=[F1Score(),
                               "accuracy",
                               Recall(name="recall"),
                               Precision(name="precision")])

        model.fit(x=dataX, y=dataY, epochs=10, validation_data=(val_dataX, val_dataY))

        test_dataX_folder = (os.path.dirname(os.path.abspath(os.path.curdir))
                             + "/Resources/TimeDataWeeks/TimeSeriesData/Week2")
        test_dataX = load_csv_from_folder(test_dataX_folder, index="timestamp").resample(RESAMPLING_RATE).mean()
        print(test_dataX)

        test_dataY_folder = (os.path.dirname(os.path.abspath(os.path.curdir))
                             + "/Resources/TimeDataWeeks/Active_phases/Week2")
        test_dataY = load_csv_from_folder(test_dataY_folder, index="timestamp").resample(RESAMPLING_RATE).median()
        print(test_dataY)

        test_dataX, test_dataY, test_index = create_dataset(dataset_X=test_dataX.loc[:, "smartMeter"],
                                                            dataset_Y=test_dataY.loc[:, "kettle"])
        print(test_dataX)
        print(test_dataY)
        print(test_index)

        print(model.evaluate(test_dataX, test_dataY))


if __name__ == '__main__':
    unittest.main()
