import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import math
from torch.nn import init
import torch.nn.functional as F
import torchvision.transforms as transforms

def get_cub_transform():
    cropsize = 448
    resize   = 512

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tflist = [transforms.ToPILImage(),transforms.RandomCrop(cropsize)]

    transform_train = transforms.Compose(
                tflist + [
                transforms.RandomRotation(15),
                transforms.RandomCrop(cropsize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ] 
                )

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        normalize
        ]
        )

    return transform_train,transform_test