from __future__ import division
import numpy as np

def precision_recall(Wtrue, D):
  """ Compute the precision and recall
      ----------------------
      -    TP   -    FP    -
      ----------------------
      -    FN   -    TN    -
      ----------------------
      True Positive  (tp): retrieved and relevant
      True Negative  (tn): not retrieved and not relevant
      False Positive (fp): retrieved and nonrelevant
      False Negative (fn): not retrieved and relevant
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)

  Args:
    Wtrue: Matrix (n_test x n_train). True neighbors
    D    : Matrix (n_test x n_train). Each element is the distance value
  Returns:
    precision and recall, both are array, with the same dimensions.
  """
  max_distance = np.amax(D)
  n_test, n_train = Wtrue.shape
  # The number of relevant data, which is obtained from the training set and
  # calculated with Euclidean distance
  tp_fn = np.sum(Wtrue)

  precision = np.zeros(max_distance)
  recall = np.zeros(max_distance)

  for n in xrange(int(max_distance)):
    # The element in matrix j with `True` value means its the retrieved data
    j = (D <= n + 0.00001)
    # Both retrieved and relevant
    tp = np.sum(Wtrue[j])
    # The number of all the retrieved data
    tp_fp = np.sum(j)
    precision[n] = tp / tp_fp
    recall[n] = tp / tp_fn

  return precision, recall
