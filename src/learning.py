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
'''resnet50
for name, param in resnet50.named_parameters():
    param.requires_grad = False
resnet50.fc = torch.nn.Sequential(
    torch.nn.Linear(resnet50.fc.in_features, 500),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(500, 2)
)
''resnet18
for name, param in resnet18.named_parameters():
    param.requires_grad = False
resnet18.fc=torch.nn.Sequential(
    torch.nn.Linear(resnet18.fc.in_features, 500),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(500, 2)
)
'''
for name, param in resnet34.named_parameters():
    param.requires_grad = False
resnet34.fc=torch.nn.Sequential(
    torch.nn.Linear(resnet34.fc.in_features, 500),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(500, 2)
)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
resnet50.to(device)
resnet18.to(device)
epochs = 5
'''resnet18
optimizer18=torch.optim.Adam(resnet18.parameters(), lr=0.001)
'''
'''resnet50
optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.001)
'''
optimizer34=torch.optim.Adam(resnet34.parameters(), lr=0.001)
loss_function = torch.nn.CrossEntropyLoss()

best_loss = 1000000
best_acc = 0
transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_path = r"datasets\\train"
test_path  = r"datasets\\test"
train_data = dataset.ImageFolder(train_path, transform)
test_data = dataset.ImageFolder(test_path, transform)
train_loader_1 = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader_1  = DataLoader(train_data, batch_size=16, shuffle=True)
best_loss = 10000000
'''resnet50
for epoch in range(epochs):
    train_loss, train_acc = train(resnet50, train_loader_1, optimizer, loss_function, device)
    test_loss, test_acc   = evaluate(resnet50, test_loader_1, loss_function, device)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')
    if test_loss < best_loss:
        torch.save(resnet50, "resnet50_best_loss.pth")
        best_loss=test_loss'''
'''resnet18
for epoch in range(epochs):
    train_loss, train_acc = train(resnet18, train_loader_1, optimizer18, loss_function, device)
    test_loss, test_acc   = evaluate(resnet18, test_loader_1, loss_function, device)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')
    if test_loss < best_loss:
        torch.save(resnet18, "resnet18_best_loss.pth")
        best_loss=test_loss'''
for epoch in range(epochs):
    train_loss, train_acc = train(resnet34, train_loader_1, optimizer34, loss_function, device)
    test_loss, test_acc   = evaluate(resnet34, test_loader_1, loss_function, device)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')
    if test_loss < best_loss:
        torch.save(resnet34, "resnet34_best_loss.pth")
        best_loss=test_loss
