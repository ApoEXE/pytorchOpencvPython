import numpy as np

matrix = np.array([[1,0,0],
                   [1,0,1],
                   [1,0,0]])
print(matrix)

edge_h = np.array([[-1,-1,-1],
                   [5,5,5],
                   [0,0,0]])

result = np.dot(matrix,edge_h)
print(result)
sum = np.sum(result)

print(sum)