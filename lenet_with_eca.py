import torch.nn as nn
import torch.nn.functional as F
import torch
import math # For log2 in ECA

class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Calculate adaptive kernel size k
        k_size = int(abs((math.log2(channel) / gamma) + (b / gamma)))
        k_size = k_size if k_size % 2 else k_size + 1 # Ensure k_size is odd
        
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x) # Output: (b, c, 1, 1)
        # Excitation: Reshape for 1D conv, apply conv, reshape back
        y = y.squeeze(-1).transpose(-1, -2) # (b, c, 1) -> (b, 1, c)
        y = self.conv(y) 
        y = y.transpose(-1, -2).unsqueeze(-1) # (b, 1, c) -> (b, c, 1) -> (b, c, 1, 1)
        
        y = self.sigmoid(y)
        return x * y.expand_as(x) # Scale

class LenetWithECA(nn.Module):
    def __init__(self, eca_gamma=2, eca_b=1):
        super().__init__()
        # 输入: (N, 3, 32, 32) for CIFAR-10, based on layer math
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)   # 输出: (N, 6, 32, 32)
        self.bn_c1 = nn.BatchNorm2d(6)
        self.eca1 = ECABlock(channel=6, gamma=eca_gamma, b=eca_b)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 输出: (N, 6, 16, 16)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # 输出: (N, 16, 12, 12)
        self.bn_c2 = nn.BatchNorm2d(16)
        self.eca2 = ECABlock(channel=16, gamma=eca_gamma, b=eca_b)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 输出: (N, 16, 6, 6)
        
        self.conv3 = nn.Conv2d(16, 120, kernel_size=6)                     # 输出: (N, 120, 1, 1)
        self.bn_c3 = nn.BatchNorm2d(120)
        self.eca3 = ECABlock(channel=120, gamma=eca_gamma, b=eca_b)
        
        self.fc1 = nn.Linear(120, 84)  
        self.bn_f1 = nn.BatchNorm1d(84)
        self.fc2 = nn.Linear(84, 10)           

    def forward(self, x):
        # Conv block 1
        out = self.conv1(x)
        out = self.bn_c1(out)
        out = F.relu(out)
        out = self.eca1(out) # Apply ECA block
        out = self.pool1(out)
        
        # Conv block 2
        out = self.conv2(out)
        out = self.bn_c2(out)
        out = F.relu(out)
        out = self.eca2(out) # Apply ECA block
        out = self.pool2(out)
        
        # Conv block 3
        out = self.conv3(out)
        out = self.bn_c3(out)
        out = F.relu(out) 
        out = self.eca3(out) # Apply ECA block
        
        # Flatten
        out = torch.flatten(out, 1) # Flatten all dimensions except batch. Output: (N, 120)
        
        # FC block 1
        out = self.fc1(out)
        out = self.bn_f1(out)
        out = F.relu(out)
        
        # Output layer
        out = self.fc2(out) # Logits layer
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