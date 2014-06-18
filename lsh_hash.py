from __future__ import division
import numpy as np

from pca import *
from compactbit import *
from manhattan_quant import *
from math import ceil

def lsh_hash(x_train, XX, nbits, manhattan_hash = False, manhattan_bit = 2):
  """
  Compute the hash code with Locality-Sensitive Hashing (LSH)

  Args:
    x_train: training data with shape (#ntest, #dimension of feature)
    XX: training and testing data with shape (#data, #dimension of feature)
    nbtis: the number of dimension of the resultant binary code
  Returns:
    Y: the compact binary code (#data, nbits)
  """
  (n_train, dim) = x_train.shape
  if manhattan_hash == True:
    nbits = int(ceil(nbits / manhattan_bit))

  Y = np.dot(XX, np.random.randn(dim, nbits))
  #print "Shape (training set) after lsh: ", Y.shape
  # Y has the size (#test x #nbits)
  if manhattan_hash == True:
    Y = manhattan_quant(Y, n_train, nbits, manhattan_bit)
  else:
    Y = Y >= 0
    Y = compactbit(Y)

  return Y
