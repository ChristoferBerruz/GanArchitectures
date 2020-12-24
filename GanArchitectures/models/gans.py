from .ganbase import Gan
from .basemodels import DConvGenerator, ConvDiscriminator, DenseGenerator, DenseDiscriminator
from utils.visiondata import VisionData
from utils.logger import Logger
from configs import ModelConfig
import torch.nn as nn
import torch
from torch.autograd import Variable

class MLPGan(Gan):

    def __init__(self, data:VisionData):
        channels = data.channels
        self.latent_space = ModelConfig.latent_space
        self.image_space = ModelConfig.image_space
        self.G = DenseGenerator(channels, self.latent_space, self.image_space, nn.Tanh())
        self.D = DenseDiscriminator(channels, self.image_space, nn.Sigmoid())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss = nn.BCELoss()
        self.name = "MLPGAN"
        self.ID = self.name + data.name
        self.logger = Logger(self.ID)
        lr = 0.0002
        weight_decay = 0.00001
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr, weight_decay=weight_decay)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr, weight_decay=weight_decay)

    def modify_images_to_model(self, images, batch_size):
        # MLP Gan needs to modify image by flatenning it
        return images.view(batch_size, -1)

    def get_z_for_model(self, batch_size, latent_space, device):
        return torch.randn(batch_size, latent_space, device=device)

    def get_labels(self, batch_size, device):
        real_labels = Variable(torch.ones(batch_size, device=device))
        fake_labels = Variable(torch.zeros(batch_size, device=device))
        return (real_labels, fake_labels)

    def reshape_samples(self, samples):
        return samples.view(samples.size(0), 1, self.image_space, self.image_space)


class DCGan(Gan):
    """Deconvolutional Gan"""

    def __init__(self, data:VisionData):
        channels = data.channels
        self.latent_space = ModelConfig.latent_space
        self.image_space = ModelConfig.image_space
        self.G = DConvGenerator(channels, self.latent_space, self.image_space, nn.Tanh())
        self.D = ConvDiscriminator(channels, self.image_space, nn.Sigmoid())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss = nn.BCELoss()
        self.name = "DCGAN"
        self.ID = self.name + data.name
        self.logger = Logger(self.ID)
        lr = 0.0002
        betas = (0.5, 0.999)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr, betas=betas)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)

    def get_z_for_model(self, batch_size, latent_space, device):
        return torch.randn(batch_size, latent_space, 1, 1, device=device)

    def modify_images_to_model(self, images, batch_size):
        return images

    def reshape_samples(self, samples):
        return samples

    def get_labels(self, batch_size, device):
        real_labels = Variable(torch.ones(batch_size, device=device))
        fake_labels = Variable(torch.zeros(batch_size, device=device))
        return (real_labels, fake_labels)