import random
import numpy as np
import pylab as pl

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from load_data import *
from pca_hash import *
from lsh_hash import *
from metrics import *
from precision_recall import *

if __name__ == '__main__':
  db             = 'data/tinyImage/' # database
  f_feats        = 'feats' # the feature file
  f_train        = 'train' # the training set file
  f_test         = 'test'  # the testing set file
  nbits          = 256      # the number of bit used to present the binary code
                           # the value is better choosing in {32, 64, 128, 256}
  ntest          = 1000    # testing scale
  method         = 'lsh'   # ['pca', 'lsh', 'itq']
  aver_neighbors = 50      # the number of neighbors to obtain the ground true
  manhattan_hash = True   # whether to use the manhattan hashing
  manhattan_bit  = 2       # map each dimension to `manhattan_bit` bits

  [feats, train, test] = load_data(db, f_feats, f_train, f_test);

  rdm = random.sample(range(len(feats)), len(feats))
  # Get test data
  test_idx = rdm[0:ntest]
  # ntest x #(dimension of feature), for GIST descriptor, the second dimension
  # is 512
  x_test = []
  for idx in test_idx:
    x_test.append(feats[idx - 1][:])

  # Get train data
  train_idx = rdm[ntest:]
  x_train = []
  for idx in train_idx:
    x_train.append(feats[idx - 1][:])
  n_train = len(x_train)

  # This step is used to find the threshold to judge one retrieved image is true
  # or false
  d_true_train = distance_matrix(x_train[1:101][:], x_train) # sample 100 to fine
                                                             # the threshold
  d_ball = np.sort(d_true_train, axis = 1)
  (r, _) = d_ball.shape
  avers = []
  for i in range(r):
    avers.append(d_ball[i][aver_neighbors])
  d_ball = np.mean(avers)

  # Scale data
  x_train = x_train / d_ball
  x_test = x_test / d_ball
  d_ball = 1
  d_true_test_train = distance_matrix(x_test, x_train)
  # w_true_test_train is the ground true matrix (#test x #train)
  # If the element in this matrix is 1, means it's the image is true, otherwise
  # false
  w_true_test_train = d_true_test_train < d_ball

  # Generate training and test split and the data matrix
  XX = np.append(x_train, x_test, axis = 0)
  (r, c) = XX.shape
  # Center the data
  means = np.mean(XX, axis = 0).reshape(1, c)
  XX = XX - np.tile(means, (r, 1))

  # Various hash method
  if method == 'pca':
    Y = pca_hash(x_train, XX, nbits, manhattan_hash, manhattan_bit)
  elif method == 'lsh':
    Y = lsh_hash(x_train, XX, nbits, manhattan_hash, manhattan_bit)

  # Each row in Y is a binary code
  B1 = Y[0:n_train][:]
  B2 = Y[n_train:][:]
  if manhattan_hash == True:
    D = distance_matrix(B2, B1)
  else:
    D = hamming_distance(B2, B1)

  # Use D and w_true_test_train to get the precision and recall
  precision, recall = precision_recall(w_true_test_train, D)

  pl.clf()
  pl.plot(recall, precision, label = 'Precision-Recall curve')
  pl.xlabel('Recall')
  pl.ylabel('Precision')
  pl.ylim([0.0, 1.0])
  pl.xlim([0.0, 1.0])
  pl.title('Precision-Recall')
  pl.legend(loc = "lower left")
  pl.show()
