import unittest
import sys
import numpy as np

sys.path.append('..')
from metrics import *

class MetricesTestCase(unittest.TestCase):
  """ Test for 'metrics.py' """

  def test_hamming_distance(self):
    X1 = np.array([[8, 2], [2, 3]])
    X2 = np.array([[3, 4], [2, 2], [1, 2]])
    dist = hamming_distance(X1, X2)
    self.assertTrue(np.sum(dist == [[5, 2, 2], [4, 1, 3]]), 6)

if __name__ == '__main__':
  unittest.main()
