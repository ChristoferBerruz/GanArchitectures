from .visiondata import VisionData
from configs import ModelConfig
from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms

class DataRepository(object):
    """
    DataRepository is a singleton in charge of dispatching vision data
    """
    _instance = None

    def __init__(self):
        if DataRepository._instance != None:
            raise Exception("DataRepository is a singleton")

        DataRepository._instance = self

    @staticmethod
    def get_instance():
        
        if DataRepository._instance == None:
            DataRepository()

        return DataRepository._instance

    def get_data(self, name:str, batch_size:int, channels:int):
        train, train_loader, test, test_loader = self._get_loader(name, "../VisionData", batch_size, channels)
        return VisionData(train, train_loader, test, test_loader, name, channels, batch_size)


    def _get_loader(self, name:str, datapath:str, batch_size:int, channels:int):
        """Get training and validation data as well as data loaders for each data
        Parameters:
            datapath: location to store data
            batch_size: desire batch size for DataLoader
            data: an available dataset to be downloaded from pytorch
            gan_type: MLP Gan needs data reshaped because it works only with Grayscale images
        Returns:
            train: dataset with added transform
            train_loader: DataLoader for training data
            test: dataset with added transform
            test_loader: DataLoader for training data
        """
        train = None
        test = None

        if name == "MNIST":
            train = torchvision.datasets.MNIST(root=datapath, train=True, download=True)
            test = torchvision.datasets.MNIST(root=datapath, train=False, download=True)
        elif name == "FASHION_MNIST":
            train = torchvision.datasets.FashionMNIST(root=datapath, train=True, download=True)
            test = torchvision.datasets.FashionMNIST(root=datapath, train=False, download=True)
        elif name == "CIFAR10":
            train = torchvision.datasets.CIFAR10(root=datapath, train=True, download=True)
            test = train = torchvision.datasets.CIFAR10(root=datapath, train=False, download=True)
        else:
            raise Exception("Invalid dataset option.")

        return self._prepare_data(train, test, name, batch_size, channels=channels)


    def _prepare_data(self, train:Dataset, test:Dataset, name:str, batch_size:int, channels: int):
        """Normalizes datasets to [-1, 1] to improve training.
        Parameters:
            train: dataset used for training
            test: dataset used for testing
            name: name of the dataset
            batch_size: batch size to use for data loader
            channels: number of channels of data
        Returns:
            train: dataset with added transform
            train_loader: DataLoader for training data
            test: dataset with added transform
            test_loader: DataLoader for training data
        """
        resize_to = ModelConfig.image_space
        transform_list = []

        if name == "CIFAR10" and channels == 1:
            # CIFAR10 is a 3 channel image, so we need to change for grayscale
            transform_list = [transforms.Grayscale()]

        image_size = train[0][0].size[0]
        if resize_to != image_size:
            transform_list = [transforms.Resize((resize_to, resize_to))] + transform_list

        
        mean = tuple([0.5]*channels)
        std = tuple([0.5]*channels)
        transform_list = transform_list + [transforms.ToTensor(), transforms.Normalize(mean,std)]

        dataset_transform = transforms.Compose(transform_list)
        train.transform = dataset_transform
        test.transform = dataset_transform
        train.transforms = torchvision.datasets.vision.StandardTransform(dataset_transform)
        test.transforms = torchvision.datasets.vision.StandardTransform(dataset_transform)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0)
        
        return train, train_loader, test, test_loader
