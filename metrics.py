from __future__ import division
import numpy as np

np.seterr(divide = 'ignore', invalid = 'ignore')

def distance_matrix(X, Y):
  """ Compute the Euclidean distance between matrices X and Y

  Args:
    X: Feature matrix (#feature x #dimension)
    Y: Feature matrix (#feature x #dimension)
  Returns:
    D: Similarity matrix based on Euclidean distance (#X_feature x #Y_feature)
  """
  X = np.array(X)
  Y = np.array(Y)
  (r1, c1) = X.shape
  (r2, c2) = Y.shape

  X1 = np.tile(np.sum(X * X, axis = 1).reshape(r1, 1), (1, r2))
  X2 = np.tile(np.sum(Y * Y, axis = 1).reshape(r2, 1), (1, r1))

  R = np.dot(X, np.transpose(Y))
  D = np.sqrt(X1 + np.transpose(X2) - 2 * R)
  return D

def hamming_distance(X1, X2):
  """ Compute the hamming distance between matrices X1 and X2
      This function can only solve the situation when per = 8, that is, each
      column of X1 or X2 has the maximum value 255

  Args:
    X1: Feature matrix (#feature x #dimension)
    X2: Feature matrix (#feature x #dimension)
  Returns:
    D: Similarity matrix based on hamming distance (#X1_feature x #X2_feature)
  """
  bit_in_char = [
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
      3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
      3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
      2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
      3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
      5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
      2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
      4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
      3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
      4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
      5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
      5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

  n1, nwords = X1.shape
  n2, _ = X2.shape

  Dh = np.zeros([n1, n2])

  for i in xrange(n1):
    for j in xrange(nwords):
      res = X1[i][j] ^ X2[:, j]
      assert res.shape[0] == n2
      for k in xrange(n2):
        assert res[k] >= 0 and res[k] <= 255
        res[k] = bit_in_char[res[k]]
      Dh[i, :] = Dh[i, :] + res

  return Dh
