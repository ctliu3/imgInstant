from __future__ import division
import numpy as np
import math

def compactbit(b, per = 8):
  """
  """
  n, nbits = b.shape
  # allocate 8 bits to store per-dimensional hash value
  nwords = int(math.ceil(nbits / per))
  cb = [[0 for _ in xrange(nwords)] for _ in xrange(n)]

  for i in xrange(n):
      for j in xrange(nbits - 1, -1, -1):
        ind = int(math.floor(j / per))
        if b[i][nbits - j - 1] == True:
          cb[i][nwords - ind - 1] = cb[i][nwords - ind - 1] | (1 << (j % per))
  return np.array(cb)

def _compactbit(b):
  n, nbits = b.shape
  cb = [0 for _ in xrange(n)]
  for i in xrange(n):
    for j in xrange(nbits - 1, -1, -1):
      if b[i][nbits - j - 1] == True:
        cb[i] = cb[i] | (1 << j)
  return np.array(cb)

#a = np.array([[True, True, True, True, True, True, True, True, True]])
a = np.array([[True, False, True]])
#compactbit(a)
print compactbit(a, 8)
print compactbit(a)
