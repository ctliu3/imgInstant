import numpy as np

def percision_recall(Wtrue, D):
  """ Compute the precision and recall

  Parmaters
  ---------
  Wtrue: matrix
         Shape (n_test, n_train), true neighbors

  D    : matrix
         Shape (n_test, n_train), each element is the distance value

  Returns
  -------
  precision, recall
  """
  max_distance = np.amax(D)
  dist_threshold = min(3, max_distance)
  n_test, n_train = Wtrue.shape
  tot_good_pairs = np.sum(Wtrue)

  precision = np.zeros([max_distance, 1])
  recall = np.zeros([max_distance, 1])
  rate = np.zeros([max_distance, 1])

  for n in xrange(max_distance):
    j = (D <= (n - 1) + 0.00001)

    #tp = np.sum()

a = np.array([[1, 2], [3, 4]])
b = np.array([[True, False], [True, True]])
#print np.sum(b)
print np.zeros([4, 1])
