from pca import *
from compactbit import *
from manhattan_quant import *

def pca_hash(x_train, XX, nbits, manhattan_hash = False):
  """
  Compute the hash code with Principal Component Analysis (PCA)

  Args:
    x_train: training data with shape (#ntest, #dimension of feature)
    XX: training and testing data with shape (#data, #dimension of feature)
    nbtis: the number of dimension of the resultant binary code
  Returns:
    Y: the compact binary code (#data, nbits)
  """
  (eigvec, _) = pca(x_train, nbits)
  Y = np.dot(XX, eigvec)
  # Y has the size (#test x #eigvalue)
  if manhattan_hash == True:
    manhattan_quant()
  else:
    Y = Y >= 0
    Y = compactbit(Y)
  return Y
