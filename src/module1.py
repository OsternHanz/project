<<<<<<< HEAD
from torch import tensor

X = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
limit = int(input())
mask=X>limit
larger_than_limit_sum = X[mask].sum()
print(larger_than_limit_sum)
=======
from torch import tensor

X = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
limit = int(input())
mask=X>limit
larger_than_limit_sum = X[mask].sum()
print(larger_than_limit_sum)
>>>>>>> c83e9f2f6453f4b5b243739948ea80753d2bd72e
