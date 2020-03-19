import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.alex_net = models.alexnet(pretrained=True).features
        self.liner    = nn.Sequential(nn.Linear(12544, 4096), nn.Sigmoid())
        self.out      = nn.Linear(4096, 1)

    def cnn(self, x):
        x = self.alex_net(x)
        x = x.view(x.size()[0],-1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        cnn1 = self.cnn(x1)
        cnn2 = self.cnn(x2)
        dis  = torch.abs(cnn1 - cnn2)
        out  = self.out(dis)
        return out

'''

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        #  VGG model - in test
        self.net       = models.vgg13(pretrained=True).features
        self.liner     = nn.Sequential(nn.Linear(32768, 4096), nn.Sigmoid())
        self.out       = nn.Linear(4096, 1)

    def cnn(self, x):
        x = self.net(x)
        x = x.view(x.size()[0],-1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        cnn1 = self.cnn(x1)
        cnn2 = self.cnn(x2)
        dis  = torch.abs(cnn1 - cnn2)
        out  = self.out(dis)
        return out
'''
