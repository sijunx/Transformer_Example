import torch
import numpy as np

a = [4, 6, 2, 1]
a.reverse()
print(a)

b = a.reverse()
print(b)

c = [3, 8, 4, 2]
c = c + [99]*5
print(c)

print("-------------")
d = np.array([6, 2, 1])
print(d)
d = d.tolist()
print(d)
