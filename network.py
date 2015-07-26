from math import exp
import random
import numpy as np

class Network():
  def __init__(self,sizes):
    self.sizes = sizes
    self.W1 = np.random.randn(sizes[0],sizes[1])
    self.W2 = np.random.randn(sizes[1],sizes[2])
    self.alpha = 0.1

  def __feedforward(self,X):
    T = np.dot(X,self.W1)
    H = sigmoid_v(T)
    U = np.dot(H,self.W2)
    O = sigmoid_v(U)
    return H,O

  def __backprop(self,A,O,H,X):
    dW2 = (A - O) * O * (1 - O) * H[:,np.newaxis] * self.alpha
    self.W2 += dW2
    dW1 = np.sum((dW2 * self.W2),axis=1) * (1 - H) * X[:,np.newaxis]
    self.W1 += dW1

  def train(self,epochs,training_data,test_data=None):
    for epoch in range(epochs):
      for X,A in training_data:
        self.__train_one(X,A)
      if test_data:
        print "%d %d/%d alpha=%f" \
          % (epoch+1,self.__evaluate(test_data),len(test_data),self.alpha)
      else:
        print "epoch %d" % (epoch+1)
      self.alpha = 0.1

  def __train_one(self,X,A):
    H,O = self.__feedforward(X)
    self.__backprop(A,O,H,X)

  def __evaluate(self,test_data):
    return sum(int(np.argmax(self.classify(X))==a) for X,a in test_data)

  def classify(self,X):
    H,O = self.__feedforward(X)
    return O


def sigmoid(x):
  return 1 / (1 + exp(-x))

sigmoid_v = np.vectorize(sigmoid)

