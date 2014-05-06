from __future__ import division
import numpy as np
import math

def compactbit(b):
  """
  """
  n, nbits = b.shape
  # allocate 8 bits to store 8-dimensional hash value
  nwords = int(math.ceil(nbits / 8))
  cb = [[0 for _ in xrange(nwords)] for _ in xrange(n)]

  for i in xrange(n):
      for j in xrange(nbits - 1, -1, -1):
        ind = int(math.floor(j / 8))
        if b[i][nbits - j - 1] == True:
          cb[i][nwords - ind - 1] = cb[i][nwords - ind - 1] | (1 << (j % 8))
  return np.array(cb)

a = np.array([[True, True, True, True, True, True, True, True, True]])
compactbit(a)
