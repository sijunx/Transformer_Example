import numpy as np
import torch

a = np.array([2, 7, 1])

b = a.sum()
print(b)

c = a.sum()/a.size
print(c)

d = torch.mean(torch.tensor(a, dtype=float))
print(d)
