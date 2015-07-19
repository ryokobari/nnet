import cPickle
import gzip
import numpy as np
from network import Network

f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()

training_data_formatted = zip(training_data[0], training_data[1])
test_data_formatted = zip(test_data[0], test_data[1])

net = Network([784,100,10])
net.train(training_data_formatted,test_data_formatted)
