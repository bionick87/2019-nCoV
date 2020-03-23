import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ALexNet
'''
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.alex_net = models.alexnet(pretrained=True).features
        self.liner    = nn.Sequential(nn.Linear(12544, 6272))
        self.out      = nn.Linear(6272, 1)
        #self.sig     = nn.Sigmoid()

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

# VGG
class SmallVGG(nn.Module):
    def __init__(self):
        super(SmallVGG, self).__init__()
        #  VGG model - in test
        net         = models.vgg13(pretrained=True)
        self.svgg        = list(net.children())[:-2][0][:-13]
        print(self.svgg)
        ##########################################################
        self.pre_weights_0 = self.svgg[0].weight      
        self.svgg[0]        = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2)  
        self.svgg[0].weight.data[:, :, :, :] = self.pre_weights_0
        ##########################################################
        self.pre_weights_2 = self.svgg[2].weight      
        self.svgg[2]        = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=2).cuda()  
        self.svgg[2].weight.data[:, :, :, :] = self.pre_weights_2
    
    
    def forward(self, x):
        return self.svgg(x)

# VGG
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        #  VGG model - in test
        self.net         = SmallVGG()
        print(self.net)
        self.liner       = nn.Sequential(nn.Linear(1048576, 4096))
        self.out         = nn.Linear(4096, 1)

    def cnn(self, x):
        x = self.net(x)
        x = x.view(x.size()[0],-1)
        print(x.size())
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        cnn1 = self.cnn(x1)
        cnn2 = self.cnn(x2)
        dis  = torch.abs(cnn1 - cnn2)
        out  = self.out(dis)
        return out


# Resnet
'''
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        resnet         = models.resnext50_32x4d(pretrained=True)
        modules        = list(resnet.children())[:-1]      # delete the last fc layer.
        self.net       = nn.Sequential(*modules)
        self.liner     = nn.Sequential(nn.Linear(2048, 1024))
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
