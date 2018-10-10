#! /user/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np
import scipy
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def solve_beta(d_matrix):
  C, n = d_matrix.shape[0], d_matrix.shape[1]
  foo = np.divide(1, np.dot(np.eye(C).repeat(C).reshape(C, C * C), np.divide(np.repeat(d_matrix, C, axis=0), np.tile(d_matrix, (C, 1)))))
  b = n / C - np.sum(foo, axis=1).reshape(C, 1)
  foo = -.5 * foo
  bar = np.divide(1, d_matrix)
  A = np.dot(foo, np.transpose(bar)) + np.diag(np.sum(np.divide(1, 2 * d_matrix), axis=1))
  beta = np.linalg.solve(A, b)
  print(np.linalg.det(A))
  return beta


def solve_alpha(d_matrix, beta):
  n, foo = d_matrix.shape[1], np.divide(1, d_matrix)
  foobar, bar = 2 - np.sum(np.multiply(beta, foo), axis=0).reshape(n, 1), np.sum(foo, axis=0).reshape(n, 1)
  alpha = np.divide(foobar, bar)
  return alpha


def update_memberships(d_matrix, beta, alpha):
  C, n = d_matrix.shape[0], d_matrix.shape[1]
  foo, bar = np.tile(beta, (1, n)), np.tile(alpha.reshape(1, n), (C, 1))
  memberships = np.multiply(np.divide(foo + bar, d_matrix), .5)
  return memberships


def update_prototypes(memberships, points, m):
  foo, bar = np.dot(np.power(memberships, m), points), np.sum(np.power(memberships, m), axis=1).reshape(memberships.shape[0], 1)
  prototypes = np.divide(foo, bar)
  return prototypes


def fcm_equalsize(points, C=2, threshold=1e-5, m=2):
  prototypes = np.array([[1, 1], [2, 2], [3, 3], [4, 4.], [5, 5.]])
  memberships_ex = np.zeros((C, points.shape[0]))
  num = 0
  while True:
    d_matrix = spatial.distance_matrix(prototypes, points)
    print(d_matrix)
    beta = solve_beta(d_matrix)
    alpha = solve_alpha(d_matrix, beta)
    memberships = update_memberships(d_matrix, beta, alpha)
    prototypes = update_prototypes(memberships, points, m)
    num += 1
    print(num)
    if np.linalg.norm(memberships - memberships_ex) < threshold:
      print(num)
      break
    else:
      memberships_ex = memberships
  return np.argmax(memberships, axis=0)


if __name__ == "__main__":
  
  centers = [[0, 5], [1, 4], [2, 3], [3, 2.], [4, 1.1]]
  data, labels_true = make_blobs(centers=centers, n_samples=100)
  points = np.array(data) * 50
  ans = (fcm_equalsize(points, C=5))
  print(ans)
  #plt.scatter(points[:, 0], points[:, 1], 15, ans * 10)
  #plt.show()
  #plt.clf()
  #plt.scatter(points[:, 0], points[:, 1], 8, labels_true * 5)
  #plt.show()

  
