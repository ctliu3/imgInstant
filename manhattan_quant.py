import numpy as np

def manhattan_quant(XX, n_train, nbits, Q):
  """
  Args:
    XX: data with shape #data x #(dimension of feature)
    n_train: #training set
    nbits: #binary code
    Q: map each dimension to Q bits
  Return:
    Y: the compact numeric code (#data, nbits). NOT binary code.
  """
  X_train = XX[1:n_train, :];

  # Assuming Q equal to 2, then range of each value will be in the range of
  # [00, 01, 10, 11], where each value is the binary number
  threshold =  np.zeros([nbits, 2**(Q - 1)])

  for i in xrange(Q):
    centers = k_maens(X_train[:, i], 2**Q);
    centers = sorted(centers);
    threshold[i, :] = 0;
    pass

def _assign():
  pass
