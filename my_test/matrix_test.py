from numpy import *

a1 = [[2, 3, 4], [5, 6, 7]]  # 列表
a2 = mat(a1)  # 矩阵
print(a2)
print('----------')
matrix([[2, 3, 4],
        [5, 6, 7]])
a3 = array(a1)  # 数组
print(a3)
print('===========')
array([[2, 3, 4],
       [5, 6, 7]])

a4 = a2.tolist()
print(a4)
print('=--------=')
[[2, 3, 4], [5, 6, 7]]
a5 = a3.tolist()
print(a5)
print('=========----------')
[[2, 3, 4], [5, 6, 7]]
a4 == a5
True
