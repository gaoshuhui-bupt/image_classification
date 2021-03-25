import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as M
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import cv2

def get_affine_transform(size1, size2): 
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    scale1 = size2[0]*1.0/size1[0]
    scale2 = size2[1]*1.0/size1[1]
    scale = min(scale1,scale2)
    # Center to Center
    src[0, :] = [size1[0]/2.0 , size1[1]/2.0]
    dst[0, :] = [size2[0]/2.0 , size2[1]/2.0]

    # Left Center to Left Center Boarder 
    src[1, :] = [0.0 , size1[1]/2.0]
    dst[1, :] = [size2[0]/2.0 - scale*size1[0]/2.0 , size2[1]/2.0]

    # Top Center to Top Center Boader
    src[2, :] = [ size1[0]/2.0, 0.0]
    dst[2, :] = [ size2[0]/2.0 , size2[1]/2.0 - scale*size1[1]/2.0 ]
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

class DataSetsLeaf(Dataset):
    def __init__(self, data_root, img_path_lst, transforms=None, num_class=5): 
        self.gt = {}
        f = open(img_path_lst)
        for line in f.readlines():
            name,label = line.strip().split()
            label = int(label)
            self.gt[name] = label
        f.close()
        self.data_root = data_root
        self.transforms = transforms
        self.nF = len(self.gt) 
        self.Size = (512,512)
        print("Load data len is ",self.nF)   

    def __len__(self):
        return self.nF

    def __getitem__(self, idx):
        name = list(self.gt.keys())[idx]
        label = self.gt[name]
        img_path = os.path.join(self.data_root,name)
        img = cv2.imread(img_path)
        img = img[:,:,::-1]
        img_w,img_h = img.shape[1],img.shape[0]
        trans_input = get_affine_transform((img_w,img_h),self.Size)
        image = cv2.warpAffine(img, trans_input, self.Size,flags=cv2.INTER_LINEAR)
        label = torch.from_numpy(np.array(label))
        if self.transforms :
            image = self.transforms(image)
        return image, label

