import torchvision.models as models
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def evaluate(model, dataloader, loss_function, device):
    epoch_acc = 0
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            predicts = model(images)
            loss = loss_function(predicts, labels)
            acc  = calculate_accuracy(predicts, labels)
            epoch_loss += loss.item()
            epoch_acc  += acc.item()
    return epoch_loss / len(dataloader),  epoch_acc / len(dataloader)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, dataloader, optimizer, loss_function, device):
    epoch_acc = 0
    epoch_loss = 0
    model.train()
    for (images, labels) in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predicts = model(images)
        loss = loss_function(predicts, labels)
        acc = calculate_accuracy(predicts, labels)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc  += acc.item()
    return epoch_loss / len(dataloader),  epoch_acc / len(dataloader)

resnet18=models.resnet18(pretrained=True)
resnet34=models.resnet34(pretrained=True)
resnet50=models.resnet50(pretrained=True)
for name, param in resnet50.named_parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
resnet50.to(device)
epochs = 5
optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.001)
loss_function = torch.nn.CrossEntropyLoss()

best_loss = 1000000
best_acc = 0
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
])
train_data = dataset.FashionMNIST(r"/datasets/train",
                                  train=True,
                                  download=True,
                                  transform=transform,
                                  )
test_data = dataset.FashionMNIST(r"/datasets/test",
                                 train=False,
                                 download=True,
                                 transform=transform
                                 )
train_loader_1 = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader_1 = DataLoader(test_data, batch_size=32, shuffle=True)
for epoch in range(epochs):
    train_loss, train_acc = train(resnet50, train_loader_1, optimizer, loss_function, device)
    test_loss, test_acc   = evaluate(resnet50, test_loader_1, loss_function, device)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')
