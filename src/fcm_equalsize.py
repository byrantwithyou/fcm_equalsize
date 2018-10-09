#! /user/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np
import scipy
import scipy.spatial as spatial
from scipy import linalg
from numpy import fromfunction


def solve_beta(d_matrix):
  C = d_matrix.shape[0]
  n = d_matrix.shape[1]
  return 1

def solve_alpha(d_matrix, beta):
  print("solve_alpha")
  return 1


def update_memberships(d_matrix, beta, alpha):
  print("solve_dfs")
  return 1


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
  d_matrix = np.array([[1, 2, 3], [4, 5, 6]])
  #print(np.tile(d_matrix, (2, 1)))
  #print(np.repeat(d_matrix, 2, axis=0))
  a = np.repeat(d_matrix, 2, axis=0)
  b = np.tile(d_matrix, (2, 1))
  print(b.shape)
  print(np.divide(a, b))
  print(np.divide(1, ))
  #print(np.divide(np.repeat(d_matrix, 2, axis=0)), np.tile(d_matrix, (2, 1)))
  #a = np.array([[1, 2], [3, 4]])
  #b = np.eye(3)
  #print(np.repeat(b, 2).reshape((3, 6)))
  #print(np.repeat(a, 2, axis=0))
  
  print()
  #fcm_equalsize(points)  

