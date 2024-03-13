<<<<<<< HEAD
import torch

w=torch.tensor([[5., 10.],[1., 2.]], requires_grad=True)
optimizer=torch.optim.SGD([w], lr=0.001)
for i in range(500):
    function=torch.prod(torch.log(torch.log(w+7)))
    function.backward()
    optimizer.step()
    optimizer.zero_grad()

print(w)
=======
import torch

w=torch.tensor([[5., 10.],[1., 2.]], requires_grad=True)
optimizer=torch.optim.SGD([w], lr=0.001)
for i in range(500):
    function=torch.prod(torch.log(torch.log(w+7)))
    function.backward()
    optimizer.step()
    optimizer.zero_grad()

print(w)
>>>>>>> c83e9f2f6453f4b5b243739948ea80753d2bd72e
