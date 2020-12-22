from torch.utils.data import DataLoader, Dataset

class VisionData(object):
    """A vision data contains information about data, train and test loaders for images
    """

    def __init__(self, train:Dataset, train_loader:DataLoader, test:Dataset, test_loader:DataLoader, name:str, channels:int, batch_size:int):
        """
        Parameters:
            train: train dataset
            train_loader: loader for train dataset
            test: test dataset
            test_loader: loader for test dataset
            name: name of data
            channels: channels that each image has
            batch_size: batch_size for the loaders
        """
        self._train = train
        self._train_loader = train_loader
        self._test = test
        self._test_loader = test_loader
        self._name = name
        self._channels = channels
        self._batch_size = batch_size

    @property
    def train(self):
        return self._train

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test(self):
        return self._test

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def name(self):
        return self._name

    @property
    def channels(self):
        return self._channels

    @property
    def batch_size(self):
        return self._batch_size