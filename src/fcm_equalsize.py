#! /user/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np
import scipy
import scipy.spatial as spatial


def solve_beta(d_matrix):
  C, n = d_matrix.shape[0], d_matrix.shape[1]
  foo = np.divide(1, np.dot(np.eye(C).repeat(C).reshape(C, C * C), np.divide(np.repeat(d_matrix, C, axis=0), np.tile(d_matrix, (C, 1)))))
  b = n / C - np.sum(foo, axis=1).reshape(C, 1)
  foo = -.5 * foo
  bar = np.divide(1, d_matrix)
  A = np.dot(foo, np.transpose(bar)) + np.diag(np.sum(np.divide(1, 2 * d_matrix), axis=1))
  beta = np.linalg.solve(A, b)
  print(A)
  print(b)
  print(beta)
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


def fcm_equalsize(points, C=2, threshold=.01, m=2):
  prototypes = np.random.random_sample((C, points.shape[1]))
  memberships_ex = np.zeros((C, points.shape[0]))
  while True:
    d_matrix = spatial.distance_matrix(prototypes, points)
    beta = solve_beta(d_matrix)
    alpha = solve_alpha(d_matrix, beta)
    memberships = update_memberships(d_matrix, beta, alpha)
    prototypes = update_prototypes(memberships, points, m)
    if np.linalg.norm(memberships - memberships_ex) < threshold:
      break
    else:
      memberships_ex = memberships
  return np.argmax(memberships, axis=0)


if __name__ == "__main__":
  A = np.array([[0.34763479, 0.44971272, 0.1523048,   0.78991553,  0.16131734,  0.242627,    0.62175637,  0.56820348,  0.59893423,  0.75974445], [0.34765918 , 0.56044309,  0.60183567,  0.20504329,  0.46174492,  0.461370520, 0.39245738,  0.04516625,  0.65495541 , 0.3004999]])
  solve_beta(A)