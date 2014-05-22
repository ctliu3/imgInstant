from __future__ import division
from pca import *
from compactbit import *
from manhattan_quant import *
from math import ceil

def pca_hash(x_train, XX, nbits, manhattan_hash = False, manhattan_bit = 2):
  """
  Compute the hash code with Principal Component Analysis (PCA)

  Args:
    x_train: training data with shape (#ntest, #dimension of feature)
    XX: training and testing data with shape (#data, #dimension of feature)
    nbtis: the number of dimension of the resultant binary code
  Returns:
    Y: the compact binary code (#data, nbits)
  """
  (n_train, _) = x_train.shape
  if manhattan_hash == True:
    nbits = int(ceil(nbits / manhattan_bit))

  (eigvec, _) = pca(x_train, nbits)
  eigvec = eigvec.real
  Y = np.dot(XX, eigvec)
  print "Shape (training set) after pca: ", Y.shape
  #print Y
  # Y has the size (#test x #eigvalue)
  if manhattan_hash == True:
    Y = manhattan_quant(Y, n_train, nbits, manhattan_bit)
  else:
    Y = Y >= 0
    Y = compactbit(Y)

  return Y
