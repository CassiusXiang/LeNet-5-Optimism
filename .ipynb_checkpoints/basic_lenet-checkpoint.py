import torch.nn as nn

class BasicLeNet(nn.Module):
    def __init__(self):
        super.__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)