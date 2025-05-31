import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: (N, 3, 224, 224)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)   # 输出: (N, 6, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 输出: (N, 6, 16, 16)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # 输出: (N, 16, 12, 12)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 输出: (N, 16, 6, 6)
        self.conv3 = nn.Conv2d(16, 120, 6)
        self.fc1 = nn.Linear(120, 84)  
        self.fc2 = nn.Linear(84, 10)           

    def forward(self, x):
        in_size = x.size(0)
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = self.conv3(out)
        out = out.view(in_size, -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(out, dim=1)
        return out
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Track loss
        running_loss += loss.item() * data.size(0)
        # Track accuracy
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
        
        if (batch_idx+1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    
    test_loss /= total
    test_acc = 100. * correct / total
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total, test_acc))
    return test_loss, test_acc

def train_model(model, device, train_loader, test_loader, optimizer, epochs, val_accuracies):
    model.to(device)  # Ensure model is on the correct device in subprocess
    for epoch in range(1, epochs + 1):
        print(f"--- Model {model.__class__.__name__}: Epoch {epoch}/{epochs} ---")
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        print(f"Model {model.__class__.__name__} Epoch {epoch} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
        val_loss, val_acc = test(model, device, test_loader)
        val_accuracies.append(val_acc)
        print(f"Model {model.__class__.__name__} Epoch {epoch} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")