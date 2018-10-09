import numpy as np
import random
from scipy.spatial import distance_matrix
random.seed()
data = []
with open("data.txt", "r") as file:
    data = [(float(line[:line.find(",")]), float(line[line.find(",") + 1: -1])) for line in file]
threshold, K = 0.01, 2
xMin, xMax, yMin, yMax = min([x[0] for x in data]), max([x[0] for x in data]), min([x[1] for x in data]), max([x[1] for x in data])
p = [(random.uniform(xMin, xMax), random.uniform(yMin, yMax)) for x in range(K)]
points = np.array(data)
