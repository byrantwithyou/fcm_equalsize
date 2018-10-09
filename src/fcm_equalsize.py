#! /user/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np
import scipy
import scipy.spatial as spatial
from scipy import linalg


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
  beta = linalg.solve(A, b)
  return beta


def solve_alpha(d_matrix, beta):
  n, foo = d_matrix.shape[1], np.divide(1, d_matrix)
  foobar, bar = 2 - np.sum(np.multiply(beta, foo), axis=0).reshape(n, 1), np.sum(foo, axis=0).reshape(n, 1)
  alpha = np.divide(foobar, bar)
  return alpha


def update_memberships(d_matrix, beta, alpha):
  C, n = d_matrix.shape[0], d_matrix.shape[1]
  foo, bar = np.tile(beta, (1, n)), np.tile(alpha.reshape(1, n), (C, 1))
  memberships = np.multiply(np.divide(foo + bar, d_matrix), 0.5)
  return memberships


def update_propotypes(memberships, points, m):
  foo, bar = np.dot(np.power(memberships, m), points), np.sum(np.power(memberships, m), axis=1).reshape(memberships.shape[0], 1)
  prototypes = np.divide(foo, bar)
  return prototypes


def fcm_equalsize(points, C=2, threshold=0.01, m=2):
  prototypes = np.random.random_sample((C, points.shape[1]))
  while False:
    d_matrix = spatial.distance_matrix(prototypes, points)
    beta = solve_beta(d_matrix)
    alpha = solve_alpha(d_matrix, beta)
    memberships = update_memberships(d_matrix, beta, alpha)
    prototypes = update_propotypes(memberships, points, m)
  return np.argmax(memberships, axis=0)

if __name__ == "__main__":
  print("Hello World!")
  
  

