from os import path
from torch.utils import tensorboard
from configs import ModelConfig

class Logger(object):

    def __init__(self, experiment_name):
        """Create a summary writer logger"""
        dir = path.join(ModelConfig.checkpoint_dir, experiment_name, "Logs")
        self.writer = tensorboard.SummaryWriter(dir)

    def log_loss(self, tag, loss, iter):
        self.writer.add_scalar(tag, loss, iter)

    def close(self):
        self.writer.close()


