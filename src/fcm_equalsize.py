#! /user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import scipy
import scipy.spatial as spatial

def fcm_equaisize(points, C=2, threshold=0.01):
  d_matrix = spatial.distance_matrix(points, points)
  
if __name__ == "__main__":
  points = np.random.random_sample((100, 2))
  fcm_equaisize(points)
  


