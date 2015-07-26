import numpy as np

class Hopfield:
  def __init__(self,patterns):
    length = len(patterns[0])
    W = np.zeros([length,length])

    for P in patterns:
      W += P * P[:,np.newaxis]

    np.fill_diagonal(W,0)
    self.W = W

  def associate(self,U):
    for i in range(10):
      U = self.f(np.dot(self.W,U))
    return U

  @np.vectorize
  def f(x):
    return 1 if x>=0 else -1

if __name__ == "__main__":

  def conv(str):
    print str
    return map(lambda c: 1 if c=="#" else -1,str)

  def re_conv(arr):
    return "".join(map(lambda c: "#" if c==1 else "_",arr))

  patterns = [
    "#_#_####_#____",
    "#_#___##_#####",
    "######_##_#___",
  ]

  print("memorize:")
  hop = Hopfield(np.array(map(conv,patterns)))

  print
  print("associate:")
  #a = hop.associate(conv("##_###_####_#_"))
  a = hop.associate(conv("#_#___####_###"))

  print
  print("remember:")
  print(re_conv(a))

