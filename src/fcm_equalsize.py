#! /user/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np
import scipy
import scipy.spatial as spatial
from scipy import linalg
from numpy import fromfunction


def solve_beta(d_matrix):
  C, n = d_matrix.shape[0], d_matrix.shape[1]
  foo = np.divide(1, np.dot(np.eye(C).repeat(C).reshape(C, C * C), np.divide(np.repeat(d_matrix, C, axis=0), np.tile(d_matrix, (C, 1)))))
  bar = np.sum(foo, axis=1).reshape(C, 1)
  b = n / C - bar
  bar = np.diag(np.sum(np.divide(1, 2 * d_matrix), axis=1))
  foobar = np.divide(1, np.tile(d_matrix.reshape(1, C * n), (C, 1)))
  foobar = np.multiply(np.multiply(np.tile(foo, (1, C)), foobar), -1)
  A = np.dot(foobar, np.repeat(np.eye(C), n, axis=0))
  bar = np.diag(np.sum(np.divide(1, 2 * d_matrix), axis=1))
  A = A - bar
  return linalg.solve(A, b)


def solve_alpha(d_matrix, beta):
  n = d_matrix.shape[1]
  foo = np.divide(1, d_matrix)
  bar = np.sum(foo, axis=0).reshape(n, 1)
  foobar = 2 - np.sum(np.multiply(beta, foo), axis=0).reshape(n, 1)
  alpha = np.divide(foobar, bar)
  return alpha


def update_memberships(d_matrix, beta, alpha):
  C, n = d_matrix.shape[0], d_matrix.shape[1]
  foo = np.tile(beta, (1, n))
  bar = np.tile(alpha.reshape(1, n), (C, 1))
  memberships = np.multiply(np.divide(foo + bar, d_matrix), 0.5)
  return memberships


def update_propotypes(memberships, points, m):
  print("heloo")
  return 1


def fcm_equalsize(points, C=2, threshold=0.01, m=2):
  prototypes, n = np.random.random_sample((C, points.shape[1])), points.shape[0]
  while False:
    d_matrix = spatial.distance_matrix(prototypes, points)
    beta = solve_beta(d_matrix)
    alpha = solve_alpha(d_matrix, beta)
    memberships = update_memberships(d_matrix, beta, alpha)
    prototypes = update_propotypes(memberships, points, m)
  return memberships


if __name__ == "__main__":
  d_matrix = np.array([[1, 2, 3], [5, 6, 7]])
  beta = np.array([[2], [3]])
  alpha = np.array([[1], [2], [3]])
  C, n = d_matrix.shape[0], d_matrix.shape[1]
  foo = np.tile(beta, (1, n))
  bar = np.tile(alpha.reshape(1, n), (C, 1))
  memberships = np.multiply(np.divide(foo + bar, d_matrix), 0.5)
  print(memberships)
