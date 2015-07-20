from math import exp
import random
import numpy as np

class Network():
  def __init__(self,sizes):
    self.sizes = sizes
    self.W1 = np.random.randn(sizes[0],sizes[1])
    self.W2 = np.random.randn(sizes[1],sizes[2])
    self.alpha = 0.8

  def __delta_w2(self,A,O,H):
    return (A - O) * O * (1 - O) * H[:,np.newaxis] * self.alpha

  def __delta_w1(self,dW2,H,X):
    return np.sum((dW2 * self.W2),axis=1) * (1 - H) * X[:,np.newaxis]

  def __feedforward(self,X):
    T = np.sum((X * self.W1.T),axis=1)
    H = sigmoid_v(T)
    U = np.sum((H * self.W2.T),axis=1)
    O = sigmoid_v(U)
    return H,O

  def __backprop(self,A,O,H,X):
    dW2 = self.__delta_w2(A,O,H)
    dW1 = self.__delta_w1(dW2,H,X)
    self.W1 += dW1
    self.W2 += dW2
    
  def train(self,training_data,test_data=None):
    for epoch in range(5):
      #random.shuffle(training_data)
      for X,a in training_data:
        A = np.zeros(self.sizes[2])
        A[a] = 1.0
        self.__train_one(X,A)
      if test_data:
        print "%d %d/%d" \
          % (epoch,self.__evaluate(test_data),len(test_data))

  def __train_one(self,X,A):
    H,O = self.__feedforward(X)
    self.__backprop(A,O,H,X)

  def __evaluate(self,test_data):
    return sum(int(self.classify(X)==a) for X,a in test_data)

  def classify(self,X):
    H,O = self.__feedforward(X)
    return np.argmax(O)


def sigmoid(x):
  return 1 / (1 + exp(-x))

sigmoid_v = np.vectorize(sigmoid)

