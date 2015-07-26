import numpy as np
from network import Network
from network_numba import NetworkNumba
from mnist_loader import MnistLoader

loader = MnistLoader()

test_data = loader.test_data_fmt()
training_data = loader.training_data_fmt()

#net = Network([784,30,10])
net = NetworkNumba([784,30,10])
net.train(10,training_data,test_data)
