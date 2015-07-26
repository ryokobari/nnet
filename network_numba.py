# coding: utf-8
from math import exp
import random
import numpy as np
from numba import double
from numba.decorators import jit

class NetworkNumba():
  def __init__(self,sizes):
    self.sizes = np.array(sizes)
    self.W1 = np.random.randn(sizes[0],sizes[1])
    self.W2 = np.random.randn(sizes[1],sizes[2])
    self.alpha = 0.1

  def train(self,epochs,training_data,test_data=None):
    sizes = self.sizes
    for epoch in range(epochs):
      for X,A in training_data:
        self.__train_one(X,A)
      if test_data:
        print "%d %d/%d alpha=%f" \
          % (epoch+1,self.__evaluate(test_data),len(test_data),self.alpha)
      else:
        print "epoch %d alpha=%f" % (epoch+1,self.alpha)
      self.alpha = 0.1

  def classify(self,X):
    H = np.zeros(self.sizes[1],dtype=np.float64)
    O = np.zeros(self.sizes[2],dtype=np.float64)

    X = np.array(X,dtype=np.float64)
    feedforward(self.sizes,X,H,O,self.W1,self.W2)
    return O

  def __train_one(self,X,A):
    X = np.array(X,dtype=np.float64)
    A = np.array(A,dtype=np.float64)
    backprop(self.sizes,A,X,self.W1,self.W2,self.alpha)

  def __evaluate(self,test_data):
    return sum(int(np.argmax(self.classify(X))==a) for X,a in test_data)


@jit('void(i8[:], f8[:], f8[:], f8[:], f8[:,:], f8[:,:])', nopython=True)
def feedforward(sizes,X,H,O,W1,W2):
  I,J,K = sizes

  for j in range(J):
    for i in range(I):
      H[j] += X[i] * W1[i][j]

  for j in range(J):
    H[j] = 1/(1 + exp(-H[j]))
    for k in range(K):
      O[k] += H[j] * W2[j][k]

  for k in range(K):
    O[k] = 1/(1 + exp(-O[k]))

@jit('void(i8[:], f8[:], f8[:], f8[:,:], f8[:,:],f8)', nopython=True)
def backprop(sizes,A,X,W1,W2,alpha):
  I,J,K = sizes
  H = np.zeros(J,dtype=np.float64)
  Z = np.zeros(J,dtype=np.float64)
  O = np.zeros(K,dtype=np.float64)

  feedforward(sizes,X,H,O,W1,W2)

  for j in range(J):
    for k in range(K):
      dw2k = (A[k] - O[k]) * O[k] * (1 - O[k]) * H[j] * alpha
      W2[j][k] += dw2k
      Z[j] += W2[j][k] * dw2k

  for i in range(I):
    for j in range(J):
      W1[i][j] += Z[j] * (1 - H[j]) * X[i]


