import torch

from data import zidian_yr
from main01 import predict

x = [0, 30, 35, 23, 17, 29, 31, 27, 34, 19, 27, 31, 29, 36, 36, 20, 35, 9,
     38, 23, 10, 36, 37, 37, 8, 29, 38, 30, 7, 36, 34, 12, 27, 22, 25, 32,
     33, 9, 28, 23, 26, 11, 36, 6, 25, 1, 2, 2, 2, 2]
x = torch.tensor(x)
x = torch.unsqueeze(x, 0)

print("预测结果---------------------------------")
y = predict(x)
print("y:", y)
print("y[0]:", y[0])
out = ' '
for i in y[0]:
     print("i:", i)
     out = out + zidian_yr[i]

print("out:", out)
