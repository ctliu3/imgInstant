import numpy as np

np.seterr(divide = 'ignore', invalid = 'ignore')
def dist_mat(x_feat1, x_feat2):
  """
  """
  #n1, n2 = len(x_feat1), len(x_feat2)
  #d = len(x_feat1[0])
  ##print n1, n2, d
  #dis = [[0] * n2 for _ in xrange(n1)]
  #for i in xrange(n1):
    #for j in xrange(n2):
      #val = 0
      #for k in xrange(d):
        #temp = abs(x_feat1[i][k] - x_feat2[j][k])
        #val += temp * temp
      #dis[i][j] = math.sqrt(val)
  #return dis

  x_feat1 = np.array(x_feat1)
  x_feat2 = np.array(x_feat2)
  (r1, c1) = x_feat1.shape
  (r2, c2) = x_feat2.shape
  print r1, c1
  print r2, c2
  x1 = np.tile(np.sum(x_feat1 * x_feat1, axis = 1).reshape(r1, 1), (1, r2))
  x2 = np.tile(np.sum(x_feat2 * x_feat2, axis = 1).reshape(r2, 1), (1, r1))
  print "x1: ", x1.shape
  print "x2: ", x2.shape
  R = np.dot(x_feat1, np.transpose(x_feat2))
  D = np.sqrt(x1 + np.transpose(x2) - 2 * R)
  return D

# test
#a = np.array([[1, 2], [1, 3]])
#print a < 10
#b = np.array([[3, 2], [1, 2], [5, 6]])
#c = np.array([[1, 2], [1, 3]])
#d = np.array([[4, 2], [9, 7]])
#print c
#print d
#print np.real(c)
#print c + d
#print 2 + c
#print np.sqrt(np.array(d))
#res = dist_mat(a, b)
#print res
#a = np.array([[1, 2], [4, 3]])
#print np.sort(a, axis = 1)
