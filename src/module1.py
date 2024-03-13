from torch import tensor

X = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
limit = int(input())
mask=X>limit
larger_than_limit_sum = X[mask].sum()
print(larger_than_limit_sum)
