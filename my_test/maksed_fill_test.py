import torch

score = torch.Tensor([[3, 4, 1],
                     [5, 3, 2],
                     [7, 8, 2]])

input = torch.Tensor([[6, 4, 2],
                      [8, 3, 6],
                      [5, 6, 2]])

mask = input == 2
print("mask:", mask)

score = score.masked_fill_(mask, -float('inf'))
print(score)









