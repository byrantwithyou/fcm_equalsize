import numpy 
import scipy
from scipy.spatial import distance_matrix
points = numpy.random.random_sample((10, 2))
prototypes = numpy.random.random_sample((2, 2))
d_matrix = numpy.array([[0.34763479, 0.44971272, 0.1523048,   0.78991553,  0.16131734,  0.242627,    0.62175637,  0.56820348,  0.59893423,  0.75974445], [
             0.34765918, 0.56044309,  0.60183567,  0.20504329,  0.46174492,  0.461370520, 0.39245738,  0.04516625,  0.65495541, 0.3004999]])
C = 2
n = 10
A = numpy.zeros((C, C))
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

b = numpy.zeros((C, 1))
b += n / C
for k in range(C):
  for j in range(n):
    foo = 0
    for i in range(C):
      foo += d_matrix[k][j] / d_matrix[i][j]
    b[k] += -1. / foo
print(A)
print(b)
print(numpy.linalg.solve(A, b))
