#! /user/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np
import scipy
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def solve_beta(d_matrix, beta_0):
  C, n = d_matrix.shape[0], d_matrix.shape[1]
  foo = np.divide(1, np.dot(np.eye(C).repeat(C).reshape(C, C * C), np.divide(np.repeat(d_matrix, C, axis=0), np.tile(d_matrix, (C, 1)))))
  b = n / C - np.sum(foo, axis=1).reshape(C, 1)
  foo = -.5 * foo
  bar = np.divide(1, d_matrix)
  A = np.dot(foo, np.transpose(bar)) + np.diag(np.sum(np.divide(1, 2 * d_matrix), axis=1))
  A, b = A[1:, 1:], b[1:, :] - beta_0 * A[1:, 0].reshape(C - 1, 1)
  beta = np.linalg.solve(A, b)
  beta = np.insert(beta, 0, beta_0, axis=0) 
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


def fcm_equalsize(points, centers, threshold=1e-5, m=2, beta_0 = 0):
  prototypes = np.array(centers)
  C = prototypes.shape[0]
  memberships_ex = np.zeros((C, points.shape[0]))
  errors = []
  while True:
    d_matrix = spatial.distance_matrix(prototypes, points)
    beta = solve_beta(d_matrix, beta_0)
    alpha = solve_alpha(d_matrix, beta)
    memberships = update_memberships(d_matrix, beta, alpha)
    prototypes = update_prototypes(memberships, points, m)
    error = np.linalg.norm(memberships - memberships_ex)
    if error < threshold:
      break
    else:
      memberships_ex = memberships
      errors.append(error)
  return np.argmax(memberships, axis=0), errors


if __name__ == "__main__":
  centers = [[0, 5], [1, 4], [2, 3], [3, 2.], [4, 1.1]]
  data, labels_true = make_blobs(centers=centers, n_samples=1000)
  points = np.array(data)
  ans = fcm_equalsize(points, centers, threshold=1e-1, m=4)
  print("迭代次数：")
  print(len(ans[1]))
  points_ans = ans[0]
  print("每类数量：") 
  stat = {}
  for x in range(len(centers)):
    stat[str(x)] = 0
  for x in points_ans:
    stat[str(x)] += 1
  print(stat)
  print("标准差：")
  std = np.std(np.array([value for key, value in stat.items()]))
  print(std)
  plt.scatter(points[:, 0], points[:, 1], 15, points_ans * 10)
  plt.show()
  plt.plot([x for x in range(len(ans[1]))], ans[1])
  plt.show()

  #plt.scatter(points[:, 0], points[:, 1], 8, labels_true * 5)
  #plt.show()

  
