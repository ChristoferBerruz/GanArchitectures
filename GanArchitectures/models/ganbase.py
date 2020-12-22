from .basemodels import DenseDiscriminator, DenseGenerator
from utils.visiondata import VisionData
from configs import ModelConfig
from utils.logger import Logger
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import utils
from torch.autograd import Variable
import os
from time import time
from abc import ABC, abstractmethod

class Gan(ABC):
    """
    GanBase is an abstract class for the original Gan architecture
    """

    @abstractmethod
    def __init__(self, channels:int):
        pass

    @abstractmethod
    def modify_images_to_model(self, images, batch_size):
        pass

    @abstractmethod
    def get_z_for_model(self, batch_size, latent_space, device):
        pass

    @abstractmethod
    def get_labels(self, batch_size, device):
        pass

    @abstractmethod
    def reshape_samples(self, samples):
        pass

    def save_checkpoint(self, epoch, generator_iter, number_of_images):
        checkpoint_dir = ModelConfig.checkpoint_dir
        if self.ID == None:
            raise Exception("Save checkpoint only to be called from training")
        folder_name = os.path.join(checkpoint_dir, self.ID)
        
        # Check if folder exists or not
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Generate images
        z = self.get_z_for_model(number_of_images, self.latent_space, self.device)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        samples = self.reshape_samples(samples)
        grid = utils.make_grid(samples)

        # Save images grid
        image_name = "generator_iter_{}.png".format(str(generator_iter).zfill(3))
        image_path = os.path.join(folder_name, image_name)
        utils.save_image(grid, image_path)
        self.save_model(folder_name)


    def save_model(self, folder_name):
        generator_file = "G.pkl"
        discriminator_file = "D.pkl"
        discriminator_path = os.path.join(folder_name, discriminator_file)
        generator_path = os.path.join(folder_name, generator_file)
        torch.save(self.D.state_dict(), discriminator_path)
        torch.save(self.G.state_dict(), generator_path)
        print("Models saved to {} and {}".format(discriminator_path, generator_path))

    def train(self, data:VisionData, epochs:int):

        self.to = time()
        generator_iter = 0
        train_loader = data.train_loader
        expected_batch_size = data.batch_size
        self.ID = self.name + data.name
        self.logger = Logger(self.ID)

        for epoch in range(epochs):
            for i, (images, _) in enumerate(train_loader):

                # Check if round number of batches
                if i == train_loader.dataset.__len__() // expected_batch_size:
                    break

                # Adapt image to environment
                images = images.to(self.device)

                # Flatten image 1,32x32 to 1024
                images = self.modify_images_to_model(images, expected_batch_size)

                z = self.get_z_for_model(expected_batch_size, self.latent_space, self.device)

                # Making variables nodes in computational graph
                images, z = Variable(images), Variable(z)

                # Getting labels for data
                real_labels, fake_labels = self.get_labels(expected_batch_size, self.device)

                # Train discriminator
                outputs = self.D(images).view(-1)
                d_loss_real = self.loss(outputs, real_labels)
                real_score = outputs

                # Compute BCELoss using fake images
                fake_images = self.G(z)
                outputs = self.D(fake_images).view(-1)
                d_loss_fake = self.loss(outputs, fake_labels)
                fake_score = outputs

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                z = self.get_z_for_model(expected_batch_size, self.latent_space, self.device)
                z = Variable(z)
                fake_images = self.G(z)
                outputs = self.D(fake_images).view(-1)

                g_loss = self.loss(outputs, real_labels)

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1
                if generator_iter % 10 == 0:
                    self.save_checkpoint(epoch, generator_iter, expected_batch_size)