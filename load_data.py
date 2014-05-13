from __future__ import print_function
import sys

def load_data(db, f_feats, f_train, f_test):
  """ Load the data.

  Args:
    db: the base path of the database
    f_feats: the file containing the features 
    f_train: the file containing the training set 
    f_test : the file containing the testing set 
  Returns:
    The database information, including the features, training and testing indexs.
    feats: matrix, #feature x #dimension
    train: array, #index of the training set 
    test : array, #index of the testing set 
  """
  feats = [] # #features x #dimension
  train = [] # index of training image
  test  = [] # index of testing iamge

  # Load features
  try:
    with open(db + f_feats + 'a') as f:
      for line in f:
        feats.append(map(float, line.strip().split()))
  except IOError:
    print('%s file load failed!\n' % (db + f_feats), file = sys.stdout)
    sys.exit(0)

  # Load training set
  try:
    with open(db + f_train) as f:
      for line in f:
        train.append(int(float(line.strip())))
  except IOError:
    print('%s file load failed!\n' % (db + f_train), file = sys.stdout)
    sys.exit(0)

  # Load testing set
  try:
    with open(db + f_test) as f:
      for line in f:
        test.append(int(float(line.strip())))
  except IOError:
    print('%s file load failed!\n' % (db + f_test), file = sys.stdout)
    sys.exit(0)

  return feats, train, test
