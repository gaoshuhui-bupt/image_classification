import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.models as models
import math
from torch.nn import init
import torch.nn.functional as F
from resnest import resnest50
class ModelResnet50(nn.Module):
    def __init__(self):
        super(ModelResnet50,self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.conv = nn.Sequential(*list(base_model.children())[:-2])
        # print(self.conv)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        feature = 2048

        self.classifier = nn.Linear(in_features=feature,out_features=5,bias=True)

    def forward(self,x):
        x = self.conv(x)
        feat = x
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x,feat

class ModelResnest50(nn.Module):
    def __init__(self):
        super(ModelResnest50,self).__init__()
        base_model = resnest50(pretrained=True)
        self.conv = nn.Sequential(*list(base_model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        feature = 2048

        self.classifier = nn.Linear(in_features=feature,out_features=5,bias=True)

    def forward(self,x):
        x = self.conv(x)
        feat = x
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x,feat
