from oop.patterns import Singleton
from models.gans import MLPGan, DCGan, WGanCP, WGanGP
from utils.datarepository import DataRepository

class ModelFactory(metaclass=Singleton):
    repo = DataRepository()

    def get_trainable_model(self, gan_model:str, dataset:str, batch_size:int):

        # There is one special case, when choosing an MLPGan and CIFAR10
        if gan_model == "MLPGAN" and dataset == "CIFAR10":
            visiondata = self.repo.get_data(dataset, batch_size, 1)
            model = MLPGan(visiondata)
            return model, visiondata

        model = None
        visiondata = None
        channels = None

        # Getting datasets
        if dataset == "CIFAR10":
            channels = 3
            visiondata = self.repo.get_data(dataset, batch_size, channels)
        if dataset == "MNIST" or dataset == "FASHION_MNIST":
            channels = 1
            visiondata = self.repo.get_data(dataset, batch_size, channels)
        if visiondata == None:
            raise Exception("Dataset is not available")

        # Getting gan model
        if gan_model == "MLPGAN":
            model = MLPGan(visiondata)
        if gan_model == "DCGAN":
            model = DCGan(visiondata)
        if gan_model == "WGANCP":
            model = WGanCP(visiondata)
        if gan_model == "WGANGP":
            model = WGanGP(visiondata)

        if model == None:
            raise Exception("Model is not available.")

        return model, visiondata