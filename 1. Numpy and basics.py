import numpy as np

arr = np.random.randint(1, 51, 10)
print(arr)

arr2 = np.random.rand(5)
print(arr2)

arr3 = np.random.randint(1,11, (4,3))
print(arr3)

print("first_row", arr3[0])
print("first_two_rows", arr3[0:2])
print("last", arr3[-1])

print(arr3.shape)
arr4 = arr3.reshape(6,2)
print(arr4)

arr3 = np.append(arr3, [[1], [2], [3], [4]], 1)
print(arr3)

arr3 = np.append(arr3, [[3,4,5,6]], 0)
print(arr3)

