from __future__ import print_function
import unittest
import sys
import numpy as np
import types

import matplotlib
import matplotlib.pyplot as plt

sys.path.append('..')
from kmeans import *

class KmeansTestCase(unittest.TestCase):
  """ Test for 'kmeans.py' """

  def test_kmeans(self):
    # Dataset comes from PRML
    # http://research.microsoft.com/en-us/um/people/cmbishop/prml/webdatasets/datasets.htm
    test_file = 'faithful.txt'
    X = []

    try:
      with open(test_file) as f:
        for line in f:
          _x, _y = map(float, line.split())
          X.append([_x, _y])
    except IOError:
      print('%s file loads failed!\n' % (test_file), file = sys.stdout)
      sys.exit(0)

    X = np.array(X)
    kmeans = Kmeans(X, 2)
    clusters = kmeans.clustering()

    matplotlib.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(clusters[0][:, 0], clusters[0][:, 1], 'bo')
    ax.plot(clusters[1][:, 0], clusters[1][:, 1], 'ro')
    plt.show()

if __name__ == '__main__':
  unittest.main()
