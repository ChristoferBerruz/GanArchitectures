from os import path

class Logger(object):
    log_dir = "../Logs/"

    def __init__(self, experiment_name):
        """Create a summary writer logger"""
        dir = path.join(log_dir, experiment_name)
        self.writer = tensorboard.SummaryWriter(dir)

    def log_loss(self, tag, loss, iter):
        self.writer.add_scalar(tag, loss, iter)

    def close(self):
        self.writer.close()


