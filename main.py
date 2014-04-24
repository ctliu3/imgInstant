#from PIL import Image
#import leargist
import random

def load_data(db, f_feats, f_train, f_test):
  feats = [] # #features x #dimension
  train = [] # index of training image
  test = []  # index of testing iamge

  # load features
  with open(db + f_feats) as f:
    for line in f:
      feats.append(map(float, line.strip().split()))
  # log

  # load training set
  with open(db + f_train) as f:
    for line in f:
      train.append(int(float(line.strip())))
  # log

  # load testing set
  with open(db + f_test) as f:
    for line in f:
      test.append(int(float(line.strip())))
  # log
  return feats, train, test

if __name__ == '__main__':
  db = 'data/tinyImage/'
  f_feats = 'feats'
  f_train = 'train'
  f_test = 'test'
  nbits = 64
  ntest = 1000 #testing scale
  method = 'pca'
  [feats, train, test] = load_data(db, f_feats, f_train, f_test);

  #
  sample_idx = random.sample(range(len(test)), ntest)
  x_test = []
  for idx in sample_idx:
    x_test.append(feats[test[idx] - 1][:])
  #print len(x_test), len(x_test[0])

  #
