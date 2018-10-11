import numpy as np
import scipy
from sklearn.datasets import make_blobs
from scipy.spatial import distance_matrix


centers = [[1, 1], [1.3, 1.1], [0.8, 0.9], [1.1, 0.], [.9, 1.1]]
data, labels_true = make_blobs(centers=centers, n_samples=100)
points = np.array(data)
prototypes = np.array(centers)
d_matrix = distance_matrix(prototypes, points) 
C = d_matrix.shape[0]
n = d_matrix.shape[1]
A = np.zeros((C, C))
for k in range(C):
  for l in range (C):
    for j in range(n):
      foo = 0
      for i in range(C):
        foo += d_matrix[k][j] / d_matrix[i][j]
      foo *= 2
      A[k][l] += -1. / (d_matrix[l][j] * foo)
      if k == l:
        A[k][l] += .5 / d_matrix[k][j]

b = np.zeros((C, 1))
b += n / C
for k in range(C):
  for j in range(n):
    foo = 0
    for i in range(C):
      foo += d_matrix[k][j] / d_matrix[i][j]
    b[k] += -1. / foo
print(np.linalg.solve(A, b))