import numpy as np
from math import ceil
from types import *
from kmeans import *
from sklearn.cluster import KMeans

def manhattan_quant(X, n_train, nbits, Q):
  """
  Args:
    X: data with shape #data x #(dimension of feature)
    n_train: #training set
    nbits: #binary code
    Q: map each dimension to Q bits
  Return:
    Y: the compact numeric code (#data, nbits). NOT binary code.
  """
  print "Manhattan quantiztion process."

  X_train = X[0:n_train, :];

  # Each Q bits present a numeric number
  # Assuming Q equal to 2, then range of each value will be in the range of
  # [00, 01, 10, 11], where each value is the binary number
  threshold =  np.zeros([nbits, 2 ** Q - 1])

  for i in xrange(nbits):
    data = np.zeros([X_train.shape[0], 1])
    for r in xrange(data.shape[0]):
      data[r, 0] = X_train[r, i]

    k_means = KMeans(n_clusters = 2 ** Q)
    k_means.fit(data)
    centers = k_means.cluster_centers_
    centers = [centers[j][0] for j in xrange(len(centers))]

    #_, centers = Kmeans(data, 2 ** Q).clustering()
    #centers = centers.tolist()
    #centers = [centers[i][0] for i in xrange(len(centers))]
    centers = sorted(centers)
    threshold[i, :] = (np.array(centers[0:-1]) + np.array(centers[1:])) / 2.0

  (n, d) = X.shape
  Y = np.zeros([n, d])
  for i in xrange(n):
    for j in xrange(d):
      quant_value = _assign(X[i, j], threshold[j, :], Q)
      assert quant_value >= 0 and quant_value < (2 ** Q)
      Y[i, j] = quant_value

  return Y

def _assign(value, threshold, Q):
  """ Assign each value to a quantization value between [0, 2 ** Q)

  Args:
    value: the numeric value needed to be quantized
    threshold: array assist to quantization
    Q: the number of bit for value after quantized

  Return:
    quantized value
  """
  assert 2 ** Q - 1 == threshold.size
  nthres = 2 ** Q

  if value < threshold[0]:
    return 0
  if value > threshold[-1]:
    return nthres - 1

  for p in xrange(nthres - 1):
    if value >= threshold[p] and value < threshold[p + 1]:
      return p + 1
