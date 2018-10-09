import random
random.seed()
file = open("data.txt", "w")
for i in range(100):
  print(random.gauss(50, 30), random.gauss(50, 30), sep=",", file=file)
