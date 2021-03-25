#!/usr/bin/env python
#coding:utf-8
import numpy as np
import os
from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as M
from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# from cnn_finetune import make_model
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import PIL
from PIL import Image
from efficientnet_pytorch import EfficientNet
import json
import cv2

# create Dataset
#img_path_lst is txt , line img_path + "\t" + label[0] + "\t" + abel[1] ......
class MyDataset(Dataset):
    def __init__(self, img_path_lst, transforms=None, isMixup=False, num_class=5):
        #self.dataset_info = dataset_info
        self.img_withLabel = []
        f = open(img_path_lst)
        line = f.readline()
        self.SIZE =(416, 416)        
        while line:
            image_path_label = line.strip().split("\t")
            tmp_c_path_label = [image_path_label[0]]
            
            for tmp in image_path_label[1:]:
                tmp_c_path_label.append(tmp)
            self.img_withLabel.append(tmp_c_path_label)
 
            line = f.readline()
            
        self.nF = len(self.img_withLabel)
        self.num_class = num_class

        self.transforms = transforms
        self.isMixup = isMixup
        if self.isMixup:
            self.pre_mixup()

    def __len__(self):
        return self.nF

    def load_image(self, path):
        image = PIL.Image.open(path)
        #print("image.size is ", image.size)
        
        org_img_wid, org_img_height = image.size[0], image.size[1]
        #print(org_img_wid, org_img_height, self.SIZE)
        trans = self.get_affine_transform( (org_img_wid, org_img_height), (512,512))
        
        
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        img = cv2.warpAffine(img, trans, self.SIZE)
        #cv2.imwrite("after_read_cv2.jpg", img)
        
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #image.save('after_read.jpg')
        #input('2')#('1')
        return image
    
    def pre_mixup(self):
        index_array = np.arange(int(len(self.dataset_info)))
        np.random.shuffle(index_array)
        for i in range(len(self.dataset_info)):
            lam = np.random.beta(0.4, 0.4)
            label = lam * self.dataset_info[i]['labels'] + (1 - lam) * self.dataset_info[index_array[i]]['labels']
            self.data_info.append([self.dataset_info[i]['path'], self.dataset_info[index_array[i]]['path'],
                                  label, lam])
            
    def mixup(self, data_info):
        x1 = self.load_image(data_info[0])
        x1 = self.transforms(x1)
        x2 = self.load_image(data_info[1])
        x2 = self.transforms(x2)
        lam = data_info[3]
        image = lam * x1 + (1 - lam) * x2
        return image
    
    
    def get_affine_transform(self, size1, size2): 
        
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

        
    def __getitem__(self, idx):
        if self.isMixup:
            label = self.data_info[idx][2]
            image = self.mixup(self.data_info[idx])
        else:
            img_name_label = self.img_withLabel[idx]
            img_dir = img_name_label[0]
            img_label = [int(x) for x in img_name_label[1:] ] 
            #print(" img_label ", img_name_label, img_label)
            
            """
            img_dir = self.dataset_info[idx]['path']
            label = self.dataset_info[idx]['labels']
            """
            image = self.load_image(img_dir)
            #image.save('before.png')
            #print("image before is", image.size)
            #pixel = image.getpixel((20, 20))   
            #print("pixel is ", pixel)
            
            image = self.transforms(image)
            
        #label_pyt = torch.from_numpy(label)#.astype(np.int8))
        #label = torch.zeros(1, self.num_class)
        #label.scatter_(1,  torch.tensor( [img_label]),  1 )
        
        label = torch.from_numpy(np.array(img_label))
        #print("image is", image.shape)
        return image, label, img_dir

   
"""
data_root = "../"    #../data_cassava_leaf/
train_path = data_root + "data_cassava_leaf/train_path_cassava.txt"
test_path = data_root + "data_cassava_leaf/test_path_cassava.txt"
show_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     transforms.RandomErasing(p=0.5, scale=[0.005, 0.01], ratio=[0.3, 3.3])
])
test = MyDataset(train_path,   transforms=show_transformer, num_class = 5, isMixup=False)
test.__getitem__(5)

#tmp = tmp.view(380,380,3)*255
#print("tmp shape is : ", tmp.shape,tmp[100,100,:])
#tmp_numpy = tmp.numpy().transpose(1, 2, 0) 
#tmp_numpy = tmp_numpy[:,:,::-1]
"""
