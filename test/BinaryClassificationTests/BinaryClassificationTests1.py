import os.path
import unittest

from keras.src.losses import binary_crossentropy
from keras.src.metrics import F1Score, Recall, Precision
from keras.src.optimizers import Adam

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

        model = create_model()
        model.compile(loss="binary_crossentropy", optimizer="adam",
                      metrics=[F1Score(),
                               "accuracy",
                               Recall(name="recall"),
                               Precision(name="precision")])

        model.fit(x=dataX, y=dataY, epochs=10)


if __name__ == '__main__':
    unittest.main()
