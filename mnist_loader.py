import cPickle
import gzip
import numpy as np

class MnistLoader():
  def __init__(self):
    f = gzip.open('mnist.pkl.gz', 'rb')
    self.training_data, self.validation_data, self.test_data = cPickle.load(f)
    f.close()

  def fmt_ans(self,a):
    A = np.zeros(10,dtype=np.float64)
    A[a] = 1.0
    return A

  def test_data_fmt(self):
    return zip(self.test_data[0],self.test_data[1])

  def training_data_fmt(self):
    return zip(self.training_data[0],map(self.fmt_ans, self.training_data[1]))


