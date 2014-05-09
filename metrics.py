from __future__ import division
import numpy as np
import math

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

def hamming_distance(X1, X2, per = 8):
  """ Compute the hamming distance between matrices X1 and X2

  Parmaters
  ---------
  X1, X2: matrix
          Feature matrices with shape (n1, d) and (n2, d) resp
  per   :

  Returns
  -------
  Dh: similarity matrix based on hamming distance
      An matrix with shape (n1, n2)
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

  n1, d = X1.shape
  n2, _ = X2.shape

  Dh = np.zeros([n1, n2])
  nwords = int(math.ceil(d / per))

  for i in xrange(n1):
    for j in xrange(nwords):
      res = X1[i][j] ^ X2[:, j]
      for val in res:
        #print val
        assert val >=0 and val <= 255
        res[i] = bit_in_char[val]
      Dh[i, :] = Dh[i, :] + res
      #print "res: ", res
    #for j in xrange(n2):
      #for k in xrange(nwords):
        #for b in xrange(per):
          #if (X1[i][k] & (1 << b)) ^ (X2[j][k] & (1 << b)) != 0:
            #Dh[i][j] = Dh[i][j] + 1
  #for i in xrange(n1):
    #for j in xrange(n2):
      #for k in xrange(nwords):
        #for b in xrange(per):
          #if (X1[i][k] & (1 << b)) ^ (X2[j][k] & (1 << b)) != 0:
            #Dh[i][j] = Dh[i][j] + 1
  return Dh

a = np.array([[4]])
b = np.array([[3, 3], [4, 5]])
c = np.array([1, 2, 3, 5, 3])
#print c[np.ix_([1, 2, 3, 4, 5])]
#print b.reshape(2, 2)
#print (b[0:2])[0]
#print a
#print b[:, 0]
#print np.bitwise_xor(a, b[:, 0])
#print np.bitwise_xor(a, b)
#print hamming_distance(a, b)
