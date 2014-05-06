import numpy as np

np.seterr(divide = 'ignore', invalid = 'ignore')

def distance_matrix(X, Y):
  """ Compute the distance between matrices X and Y

  Parmaters
  ---------
  X: matrix
     A feature matrix with shape (n, d)

  Y: matrix
     An feature matrix with shape (m, d)

  Returns
  -------
  D: similarity matrix
     An matrix with shape (n, m)
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

def manhttan_distance(XX, n_train, nbits, q):
  """
  """
  pass

def hamming_distance(X1, X2):
  """
  """
  n1, d = X1.shape
  n2, _ = X2.shape

  Dh = np.zeros([n1, n2])
  print Dh.shape
  return Dh

a = np.array([[23, 1, 9], [233, 9, 10]])
b = np.array([[3, 10, 7], [233, 9, 10]])
hamming_distance(a, b)
