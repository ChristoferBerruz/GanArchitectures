import argparse
import os

def get_parser() -> argparse.ArgumentParser:
    """Custom parser for Gan models
    """
    parser = argparse.ArgumentParser(description="Python implementation of multiple GAN models")

    parser.add_argument('--model', type=str, default='DCGAN', choices=['MLPGAN', 'DCGAN', 'WGANCP', 'WGANGP'])
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST', 'FASHION_MNIST'])
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of the batch')
    parser.add_argument('--generator_iter', type=int, default=1000, help='Iterations for generator. Only used fro WGAN types')
    parser.add_argument('--resume_training', type=bool, default=True, help='Whether to resume training or not')
    return parser