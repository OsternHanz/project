<<<<<<< HEAD
import torch

def loss(pred, target):
    squares=(pred-target)**2
    return squares.mean()

def target_function(x):
    return 2**x * torch.sin(2**-x)

class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(RegressionNet, self).__init__()
        self.fc1=torch.nn.Linear(1, n_hidden_neurons)
        self.act1=torch.nn.Tanh()
        self.fc2=torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x=self.fc1(x)
        x=self.act1(x)
        x=self.fc2(x)
        return x

net = RegressionNet(15)

# ------Dataset preparation start--------:
x_train =  torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)
# ------Dataset preparation end--------:


optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
for epoch_index in range(2000):
    optimizer.zero_grad()
    y_pred = net.forward(x_train)
    loss_value = loss(y_pred, y_train)
    loss_value.backward()
    optimizer.step()
    # make backward
    # make step

#Проверка осуществляется вызовом кода:
def metric(pred, target):
    return (pred - target).abs().mean()

print(metric(net.forward(x_validation), y_validation).item())
# (раскомментируйте, если решаете задание локально)
=======
import torch

def loss(pred, target):
    squares=(pred-target)**2
    return squares.mean()

def target_function(x):
    return 2**x * torch.sin(2**-x)

class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(RegressionNet, self).__init__()
        self.fc1=torch.nn.Linear(1, n_hidden_neurons)
        self.act1=torch.nn.Tanh()
        self.fc2=torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x=self.fc1(x)
        x=self.act1(x)
        x=self.fc2(x)
        return x

net = RegressionNet(15)

# ------Dataset preparation start--------:
x_train =  torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)
# ------Dataset preparation end--------:


optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
for epoch_index in range(2000):
    optimizer.zero_grad()
    y_pred = net.forward(x_train)
    loss_value = loss(y_pred, y_train)
    loss_value.backward()
    optimizer.step()
    # make backward
    # make step

#Проверка осуществляется вызовом кода:
def metric(pred, target):
    return (pred - target).abs().mean()

print(metric(net.forward(x_validation), y_validation).item())
# (раскомментируйте, если решаете задание локально)
>>>>>>> c83e9f2f6453f4b5b243739948ea80753d2bd72e
