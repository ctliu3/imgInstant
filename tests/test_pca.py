import unittest
import sys
import numpy as np

sys.path.append('..')
from pca import *

class PcaTestCase(unittest.TestCase):
  """ Test for 'pca.py' """

  def test_pca(self):
    X = np.array([
      [2.5, 2.4],
      [0.5, 0.7],
      [2.2, 2.9],
      [1.9, 2.2],
      [3.1, 3.0],
      [2.3, 2.7],
      [2.0, 1.6],
      [1.0, 1.1],
      [1.5, 1.6],
      [1.1, 0.9],
      ]);
    # The eigenvalues is [0.49, 1.28] and the eigenvectors is
    # [[-.725, -.678], [.678, -.735]]
    # Error!
    # eigvec, eigval = pca(X, 2)

if __name__ == '__main__':
  unittest.main()
