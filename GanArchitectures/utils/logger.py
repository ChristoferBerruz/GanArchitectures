from os import path
from torch.utils import tensorboard
from configs import ModelConfig

class Logger(object):

    def __init__(self, experiment_name):
        """Create a summary writer logger"""
        log_dir = path.join(ModelConfig.checkpoint_dir, experiment_name, "Logs")
        self.writer = tensorboard.SummaryWriter(log_dir)

    def log_loss(self, tag, loss, iteration):
        self.writer.add_scalar(tag, loss, iteration)

    def close(self):
        self.writer.close()


