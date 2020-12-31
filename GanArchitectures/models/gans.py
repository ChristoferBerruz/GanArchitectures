from .ganbase import MLPGanAbc, DCGanAbc, WGanAbc
from .basemodels import DConvGenerator, ConvDiscriminator, DenseGenerator, DenseDiscriminator, ConvDiscriminatorInstanceNorm
from utils.visiondata import VisionData
from utils.logger import Logger
from configs import ModelConfig
import torch.nn as nn
import torch
from torch import autograd
from torch.autograd import Variable

class MLPGan(MLPGanAbc):

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


class DCGan(DCGanAbc):
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


class WGanCP(WGanAbc):

    def __init__(self, data):
        channels = data.channels
        self.latent_space = ModelConfig.latent_space
        self.image_space = ModelConfig.image_space
        self.G = DConvGenerator(channels, self.latent_space, self.image_space, nn.Tanh())
        self.D = ConvDiscriminator(channels, self.image_space, nn.Identity())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = "WGANCP"
        self.ID = self.name + data.name
        self.logger = Logger(self.ID)
        lr = 0.00005
        self.weight_clipping_limit = 0.01
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=lr)
        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=lr)
        self.critic_iter = 5

    def one_minus_one(self):
        one = torch.FloatTensor([1])
        minues_one = -1*one
        return (one, minues_one)

    def gradient_penalty(self, real_images, fake_images, batch_size):
        return None

    def clamp(self):
        for p in self.D.parameters():
            p.data.clamp_(-self.weight_clipping_limit, self.weight_clipping_limit)


class WGanGP(WGanAbc):

    def __init__(self, data):
        channels = data.channels
        self.latent_space = ModelConfig.latent_space
        self.image_space = ModelConfig.image_space
        self.G = DConvGenerator(channels, self.latent_space, self.image_space, nn.Tanh())
        self.D = ConvDiscriminatorInstanceNorm(channels, self.image_space, nn.Identity())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = "WGANGP"
        self.ID = self.name + data.name
        self.logger = Logger(self.ID)
        lr = 1e-4
        betas = (0.5, 0.999)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr, betas=betas)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        self.critic_iter = 5
        self.lambda_term = 10

    def one_minus_one(self):
        one = torch.FloatTensor([1])
        minus_one = -1*one
        return (minus_one, one)

    def gradient_penalty(self, real_images, fake_images, batch_size):
        eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta = eta.to(self.device)

        interpolated = eta*real_images + ((1-eta)*fake_images)
        interpolated = interpolated.to(self.device)
        interpolated = Variable(interpolated, requires_grad=True)

        prob_interpolated = self.D(interpolated)

        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]

        penalty  = ((gradients.norm(2, dim=1) - 1)**2).mean()*self.lambda_term
        return penalty

    def clamp(self):
        return