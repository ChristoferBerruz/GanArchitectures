import torch.nn as nn

class DenseGenerator(nn.Module):
    """Generator as a multilayer perceptron
    """

    def __init__(self, channels:int, latent_space:int, image_space:int, output_function:nn.Module,  gpu_cores:int = 0):
        """Constructor
        Parameters:
            channels: number of channels of input image, 3 if color image
            gpu_cores: number of gpu's available in environment
            latent_space: size of latent space. Usually 100
            image_space: a square image of size (image_space)x(image_space)
            output_function: last activation function of network
        """
        super().__init__()
        self.channels = channels
        self.gpu_cores = gpu_cores
        self.latent_space = latent_space
        self.main_module = nn.Sequential(
                nn.Linear(latent_space, image_space*8),
                nn.LeakyReLU(0.2, True),

                nn.Linear(image_space*8, image_space*16),
                nn.LeakyReLU(0.2, True),

                nn.Linear(image_space*16, image_space*32),
                nn.LeakyReLU(0.2, True),
            )
        
        self.output_function = output_function

    def forward(self, x):
        x = self.main_module(x)
        x = self.output_function(x)
        return x


class DenseDiscriminator(nn.Module):
    """Discriminator as a multilayer perceptron.
    """

    def __init__(self, channels:int, image_space:int, output_function:nn.Module,  gpu_cores:int = 0):
        """Constructor
        Parameters:
            channels: number of channels of input image, 3 if color image
            gpu_cores: number of gpu's available in environment
            image_space: a square image of size (image_space)x(image_space)
            output_function: last activation function of network
        """
        super().__init__()
        
        self.gpu_cores = gpu_cores
        self.channels = channels
        self.main_module = nn.Sequential(
            nn.Linear(image_space*32, image_space*16),
            nn.LeakyReLU(0.2, True),

            nn.Linear(image_space*16, image_space*8),
            nn.LeakyReLU(0.2, True),

            nn.Linear(image_space*8, 1)
        )
        
        self.output_function = output_function

    def forward(self, x):
        x = self.main_module(x)
        x = self.output_function(x)
        return x


class DConvGenerator(nn.Module):
    """
        Generator that uses strided convolutions
    """
    def __init__(self, channels:int, latent_space:int, image_space:int, output_function:nn.Module,  gpu_cores:int = 0):
        """Constructor
        Parameters:
            channels: number of channels of input image, 3 if color image
            gpu_cores: number of gpu's available in environment
            latent_space: size of latent space. Usually 100
            image_space: a square image of size (image_space)x(image_space)
            output_function: last activation function of network
        """
        super().__init__()
        self.channels = channels
        self.gpu_cores = gpu_cores
        self.latent_space = latent_space
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(latent_space, image_space*32, 4, 1, 0),
            nn.BatchNorm2d(image_space*32),
            nn.ReLU(True),
            # upsized image space = 4
            nn.ConvTranspose2d(image_space*32, image_space*16, 4, 2, 1),
            nn.BatchNorm2d(image_space*16),
            nn.ReLU(True),
            # upsized image space = 8
            nn.ConvTranspose2d(image_space*16, image_space*8, 4, 2, 1),
            nn.BatchNorm2d(image_space*8),
            nn.ReLU(True),
            # upsized image space = 16
            nn.ConvTranspose2d(image_space*8, channels, 4, 2, 1)
        )

        self.output_function = output_function

    def forward(self, x):
        x = self.main_module(x)
        x = self.output_function(x)
        return x


class ConvDiscriminator(nn.Module):
    """
        Discriminator that uses convolutional layers rather than fully connected
    """
    def __init__(self, channels:int, image_space:int, output_function:nn.Module,  gpu_cores:int = 0):
        """Constructor
        Parameters:
            channels: number of channels of input image, 3 if color image
            gpu_cores: number of gpu's available in environment
            latent_space: size of latent space. Usually 100
            image_space: a square image of size (image_space)x(image_space)
            output_function: last activation function of network
        """
        
        super().__init__()
        self.channels = channels
        self.gpu_cores = gpu_cores

        self.main_module = nn.Sequential(
            nn.Conv2d(channels, image_space*8, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(image_space*8, image_space*16, 4, 2, 1),
            nn.BatchNorm2d(image_space*16),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(image_space*16, image_space*32, 4, 2, 1),
            nn.BatchNorm2d(image_space*32),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(image_space*32, 1, 4, 1, 0)

        )

        self.output_function = output_function

    def forward(self, x):
        x = self.main_module(x)
        x = self.output_function(x)
        return x

class ConvDiscriminatorInstanceNorm(nn.Module):
    """
        Convolutional discriminator that uses InstaceNorm rather than BatchNorm.
    """
    def __init__(self, channels:int, image_space:int, output_function:nn.Module,  gpu_cores:int = 0):
        """Constructor
        Parameters:
            channels: number of channels of input image, 3 if color image
            gpu_cores: number of gpu's available in environment
            latent_space: size of latent space. Usually 100
            image_space: a square image of size (image_space)x(image_space)
            output_function: last activation function of network
        """
        
        super().__init__()
        self.channels = channels
        self.gpu_cores = gpu_cores

        self.main_module = nn.Sequential(
            nn.Conv2d(channels, image_space*8, 4, 2, 1),
            nn.InstanceNorm2d(image_space*8, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(image_space*8, image_space*16, 4, 2, 1),
            nn.InstanceNorm2d(image_space*16, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(image_space*16, image_space*32, 4, 2, 1),
            nn.InstanceNorm2d(image_space*32, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(image_space*32, 1, 4, 1, 0)

        )

        self.output_function = output_function

    def forward(self, x):
        x = self.main_module(x)
        x = self.output_function(x)
        return x