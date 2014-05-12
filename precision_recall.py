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

  Parmaters
  ---------
  Wtrue: matrix (n_test x n_train)
         True neighbors

  D    : matrix (n_test x n_train)
         Each element is the distance value

  Returns
  -------
  precision, recall, rate
  """
  max_distance = np.amax(D)
  n_test, n_train = Wtrue.shape
  tp_fn = np.sum(Wtrue)

  precision = np.zeros(max_distance)
  recall = np.zeros(max_distance)
  rate = np.zeros(max_distance)

  for n in xrange(int(max_distance)):
    j = (D <= (n - 1) + 0.00001)
    tp = np.sum(Wtrue[j])
    tp_fp = np.sum(j)
    precision[n] = tp / tp_fp
    recall[n] = tp / tp_fn
    rate[n] = tp_fn / (n_test * n_train)
    print precision[n], recall[n], rate[n]

  return precision, recall, rate
