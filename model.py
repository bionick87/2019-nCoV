import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

'''
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

'''
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        #  VGG model - in test
        resnet         = models.resnext50_32x4d(pretrained=True)
        modules        = list(resnet.children())[:-1]      # delete the last fc layer.
        self.net       = nn.Sequential(*modules)
        self.liner     = nn.Sequential(nn.Linear(2048, 1024), nn.Sigmoid())
        self.out       = nn.Linear(1024, 1)

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



class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        #  VGG model - in test
        resnet         = models.inception_v3(pretrained=True)
        modules        = list(resnet.children())[:-1]      # delete the last fc layer.
        self.net       = nn.Sequential(*modules)
        self.liner     = nn.Sequential(nn.Linear(1000, 512), nn.Sigmoid())
        self.out       = nn.Linear(512, 1)

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
