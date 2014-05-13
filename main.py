import random
import numpy as np
import pylab as pl

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from load_data import *
from metrics import *
from pca import *
from compactbit import *
from precision_recall import *

if __name__ == '__main__':
  db             = 'data/tinyImage/'
  f_feats        = 'feats'
  f_train        = 'train'
  f_test         = 'test'
  nbits          = 64
  ntest          = 1000  # testing scale
  method         = 'pca' # ['pca', 'lsh', 'itq']
  aver_neighbors = 50

  [feats, train, test] = load_data(db, f_feats, f_train, f_test);
  exit(0)

  rdm = random.sample(range(len(feats)), len(feats))
  # Get test data
  test_idx = rdm[0:ntest]
  x_test = []
  for idx in test_idx:
    x_test.append(feats[idx - 1][:])

  # get train data
  train_idx = rdm[ntest:]
  x_train = []
  for idx in train_idx:
    x_train.append(feats[idx - 1][:])
  n_train = len(x_train)

  # Define ground-truth neighbors
  d_true_train = distance_matrix(x_train[1:101][:], x_train)
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
  # This is the ground true matrix (#test x #train)
  w_true_test_train = d_true_test_train < d_ball

  # Generate training and test split and the data matrix
  XX = np.append(x_train, x_test, axis = 0)
  (r, c) = XX.shape
  means = np.mean(XX, axis = 0).reshape(1, c)
  XX = XX - np.tile(means, (r, 1))

  #
  if method == 'pca':
    (eigvec, _) = pca(x_train, nbits)
    Y = np.dot(XX, eigvec)
    # Y has the size (#test x #eigvalue)
    Y = Y >= 0
    Y = compactbit(Y)
    print "Y.shape = ", Y.shape
  elif method == 'lsh':
    pass

  # Each row in Y is a binary code
  B1 = Y[0:n_train][:]
  B2 = Y[n_train:][:]
  D = hamming_distance(B2, B1)

  # Use D and w_true_test_train to get the precision and recall
  precision, recall = precision_recall(w_true_test_train, D)

  pl.clf()
  pl.plot(recall, precision, label = 'Precision-Recall curve')
  pl.xlabel('Recall')
  pl.ylabel('Precision')
  pl.ylim([0.0, 1.05])
  pl.xlim([0.0, 1.0])
  pl.title('Precision-Recall')
  pl.legend(loc="lower left")
  pl.show()
