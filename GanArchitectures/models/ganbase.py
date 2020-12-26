from utils.visiondata import VisionData
from configs import ModelConfig
import torch
from torchvision import utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
from time import time
from abc import ABC, abstractmethod

class Gan(ABC):
    """
    GanBase is an abstract class for the original Gan architecture
    """

    @abstractmethod
    def __init__(self, data:VisionData):
        pass

    @abstractmethod
    def modify_images_to_model(self, images, batch_size):
        pass

    @abstractmethod
    def get_z_for_model(self, batch_size, latent_space, device):
        pass

    def get_labels(self, batch_size, device):
        real_labels = Variable(torch.ones(batch_size, device=device))
        fake_labels = Variable(torch.zeros(batch_size, device=device))
        return (real_labels, fake_labels)

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
        self.save_model()


    def save_model(self):
        folder_name = os.path.join(ModelConfig.checkpoint_dir, self.ID)
        generator_file = "G.pkl"
        discriminator_file = "D.pkl"
        discriminator_path = os.path.join(folder_name, discriminator_file)
        generator_path = os.path.join(folder_name, generator_file)
        torch.save(self.D.state_dict(), discriminator_path)
        torch.save(self.G.state_dict(), generator_path)
        print("Models saved to {} and {}".format(discriminator_path, generator_path))
    
    def load_model(self):
        folder_name = os.path.join(ModelConfig.checkpoint_dir, self.ID)
        discriminator_path = os.path.join(folder_name, "D.pkl")
        generator_path = os.path.join(folder_name, "G.pkl")
        self.D.load_state_dict(torch.load(discriminator_path))
        self.G.load_state_dict(torch.load(generator_path))
        print("Sucessfully loaded models from checkpoint")

    def train(self, data:VisionData, epochs:int, resume_training:bool=True):

        to = time()
        generator_iter = 0
        train_loader = data.train_loader
        expected_batch_size = data.batch_size
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)

        if resume_training:
            try:
                self.load_model()
            except Exception:
                print("Failed to load model. Training from scratch")
        else:
            print("Training from scratch")

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
                if generator_iter % 1000 == 0:
                    self.save_checkpoint(epoch, generator_iter, expected_batch_size)

                if ((i+1)%100) == 0:
                    print("Epoch: {:2d}, batch_number/all_batches: {:d}/{:d}, D_loss: {:.8f}, G_loss : {:.8f}".format(
                        epoch+1, i+1, train_loader.dataset.__len__()//expected_batch_size, d_loss.data, g_loss.data))
                    self.logger.log_loss('Loss/Discriminator', d_loss.data, generator_iter)
                    self.logger.log_loss('Loss/Generator', g_loss.data, generator_iter)

        t = time() - to
        self.logger.close()
        print("Total training time: {}".format(t))
        self.save_model()


class MLPGanAbc(Gan, ABC):

    @abstractmethod
    def __init__(self, data):
        pass

    def modify_images_to_model(self, images, batch_size):
        # MLP Gan needs to modify image by flatenning it
        return images.view(batch_size, -1)

    def get_z_for_model(self, batch_size, latent_space, device):
        return torch.randn(batch_size, latent_space, device=device)

    def reshape_samples(self, samples):
        return samples.view(samples.size(0), 1, self.image_space, self.image_space)


class DCGanAbc(Gan, ABC):

    @abstractmethod
    def __init__(self, data):
        pass

    def get_z_for_model(self, batch_size, latent_space, device):
        return torch.randn(batch_size, latent_space, 1, 1, device=device)

    def modify_images_to_model(self, images, batch_size):
        return images

    def reshape_samples(self, samples):
        return samples


class WGanAbc(DCGanAbc, ABC):

    @abstractmethod
    def __init__(self, data):
        pass

    def data_provider(self, train_loader:DataLoader):
        while True:
            for i, (images, _) in enumerate(train_loader):
                yield images

    @abstractmethod
    def gradient_penalty(self, real_images, fake_images, batch_size):
        pass

    @abstractmethod
    def one_minus_one(self):
        pass

    def train(self, data, epochs, resume_training=True):
        to = time()
        self.D = self.D.to(self.device)
        self.G = self.G.to(self.device)
        infinite_data = self.data_provider(data.train_loader)
        expected_batch_size = data.batch_size

        one, minus_one = self.one_minus_one()
        one, minus_one = one.to(self.device), minus_one.to(self.device)
        generator_iterations = epochs

        iter_num = 0

        if resume_training:
            try:
                self.load_model()
            except Exception:
                print("Failed to load model. Training from scratch")
        else:
            print("Training from scratch")

        for g_iter in range(generator_iterations):

            for p in self.D.parameters():
                p.requires_grad = True

            W_distance = 0
            d_loss = 0
            for d_iter in range(self.critic_iter):

                iter_num += 1
                self.D.zero_grad()

                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_clipping_limit, self.weight_clipping_limit)

                images = infinite_data.__next__()
                if (images.size(0) != expected_batch_size):
                    continue

                images = images.to(self.device)

                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean(0).view(1)
                d_loss_real.backward(one)

                # Train with fake images
                z = self.get_z_for_model(expected_batch_size, self.latent_space, self.device)

                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean(0).view(1)
                d_loss_fake.backward(minus_one)

                grad_penalty = self.gradient_penalty(images, fake_images, expected_batch_size)
                if grad_penalty != None:
                    grad_penalty.backward()

                grad_penalty = grad_penalty if grad_penalty != None else 0

                d_loss = d_loss_fake - d_loss_real + grad_penalty
                W_distance = d_loss_real - d_loss_fake

                self.d_optimizer.step()

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False

            self.G.zero_grad()

            z = self.get_z_for_model(expected_batch_size, self.latent_space, self.device)
            z = Variable(z)
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean().mean(0).view(1)
            g_loss.backward(one)

            g_cost = -g_loss
            self.g_optimizer.step()

            if g_iter % 1000 == 0:
                self.save_checkpoint(g_iter, g_iter, expected_batch_size)

                epoch = iter_num // (data.train_loader.dataset.__len__() // expected_batch_size)
                print("epoch: {:2d}, batches passed: {:2d}, D_loss: {:.8f}, G_loss : {:.8f}".format(
                    epoch+1, iter_num, W_distance.data[0], g_loss.data[0]))
                self.logger.log_loss('Loss/Discriminator', d_loss.data[0], g_iter)
                self.logger.log_loss('Loss/Generator', g_loss.data[0], g_iter)
                self.logger.log_loss('Loss/Wass', W_distance.data[0], g_iter)

        t = time() - to
        self.logger.close()
        print("Total training time: {}".format(t))
        self.save_model()