from __future__ import division
from random import sample
from math import sqrt
import numpy as np

class Kmeans(object):
  """K-means clustring"""

  def __init__(self, X, k):
    """ Initialization
    
    Args:
      X: data to cluster, #feature x #dimension
      k: #cluster
    Return:
      centers: matrix, cluster[i] means the data in the center i, which also a 
               matrix
    """
    self.X = X
    self.n, self.d = X.shape # n: #dataset, d: #dimension
    self.ncluster = k
    self.threshold = 1E-3

    assert self.ncluster > 0 and self.n >= self.ncluster

  def clustering(self):
    """ Main process of k-means algorithm """
    centers = np.zeros((self.n, self.d)) # store the centers
    cluster_id = np.array([0] * self.n)  # record the ID of each center
    niter = 0   # #iteration
    diff = 1e10 # to judge whether two adjcent interations have been converged

    center_id = self._init_center()
    for i in xrange(self.ncluster):
      centers[i, :] = self.X[center_id[i], :]

    while diff > self.threshold:
      # Find the nearest center for each data point
      for i in xrange(self.n):
        min_distance, close_id = 1e10, 1
        for j in xrange(self.ncluster):
          distance = self._distance(self.X[i], centers[j])
          if distance < min_distance:
            min_distance = distance
            close_id = j
        cluster_id[i] = close_id

      cluster_count = np.array([0] * self.ncluster)
      pre_centers = centers.copy()
      centers = np.zeros((self.n, self.d))

      # Calcuate the new clusters
      for i in xrange(self.n):
        centers[cluster_id[i]] += self.X[i]
        cluster_count[cluster_id[i]] += 1
      centers = np.array(\
          [centers[i] / cluster_count[i] for i in xrange(self.ncluster)])

      # Calcuate the sum of ditance of centers in two adjcent iterations
      diff = sum([self._distance(pre_centers[i], centers[i]) \
          for i in xrange(self.ncluster)])

      niter += 1

    # End of the body of k-means algorithm, return the data points in each cluster
    cluster = [[] for _ in xrange(self.ncluster)]
    [cluster[cluster_id[i]].append(self.X[i]) for i in xrange(self.n)]
    for i in xrange(self.ncluster):
      cluster[i] = np.matrix(cluster[i])
    return cluster

  def _init_center(self):
    """ Rondomly selecting self.ncluster number from [1...self.n] """
    return sample([i + 1 for i in xrange(self.n)], self.ncluster)

  def _distance(self, x, y):
    """ Calculate the distance between data x and y """
    return sqrt(sum((x - y) ** 2))
