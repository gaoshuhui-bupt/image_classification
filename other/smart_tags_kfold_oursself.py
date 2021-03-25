# coding=utf-8
import numpy as np
import os
#from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as M
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import PIL
from PIL import Image
from efficientnet_pytorch import EfficientNet

from models import model
import utils

import json
import my_dataset
import sys

import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import random


from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from skimage import io
from datetime import datetime
import time
import random
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)


from albumentations.pytorch import ToTensorV2



#mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]
CFG = {
    'num_classes':5,
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'se-resnext50',
    'img_size': 512,
    'epochs': 15,
    'train_bs': 12,
    'valid_bs': 32,
    'T_0': 15,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-5,
    'num_workers': 4,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:2'
}

device_all = CFG['device']


# implementations reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
# paper - https://www.ijcai.org/Proceedings/2020/0305.pdf

class TaylorSoftmax(nn.Module):

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

class LabelSmoothingLoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1): 
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
    def forward(self, pred, target): 
        """Taylor Softmax and log are already applied on the logits"""
        #pred = pred.log_softmax(dim=self.dim) 
        
        with torch.no_grad(): 
            #true_dist = target.clone()
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    

class TaylorCrossEntropyLoss(nn.Module):

    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.1):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(5, smoothing=smoothing)

    def forward(self, logits, labels):

        log_probs = self.taylor_softmax(logits).log()
        #print("log_probs shape is ", log_probs.shape)
        #loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss
    
    

def get_img(path):
    im_bgr = cv2.imread(path)
    #image = Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     transforms.RandomErasing(p=0.5, scale=[0.005, 0.01], ratio=[0.3, 3.3])
])

valid_transformer = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transformer = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


def get_train_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(rotate_limit=30, p=0.5),
            HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():

    return Compose([
            #CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(CFG['img_size'], CFG['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

class CassavaDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 do_fmix=False, 
                 fmix_params={
                     'alpha': 1., 
                     'decay_power': 3., 
                     'shape': (CFG['img_size'], CFG['img_size']),
                     'max_soft': True, 
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['label'].values
            #print(self.labels)
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.labels[index]
          
        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))
        #print("self.df.loc[index]['image_id'] is ", self.df.loc[index]['image_id'])

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                #lam, mask = sample_mask(**self.fmix_params)
                
                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
                
                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])
    
                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)
                
                # mix image
                img = mask_torch*img+(1.-mask_torch)*fmix_img

                #print(mask.shape)

                #assert self.output_label==True and self.one_hot_label==True

                # mix target
                rate = mask.sum()/CFG['img_size']/CFG['img_size']
                target = rate*target + (1.-rate)*self.labels[fmix_ix]
                #print(target, mask, img)
                #assert False
        
        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']
                    
                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((CFG['img_size'], CFG['img_size']), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (CFG['img_size'] * CFG['img_size']))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]
                
            #print('-', img.sum())
            #print(target)
            #assert False
                            
        # do label smoothing
        #print(type(img), type(target))
        
        #print("target",  target)
        if self.output_label == True:
            return img, target
        else:
            return img,self.df.loc[index]['image_id']

def prepare_dataloader(df, trn_idx, val_idx, data_root = '../data_cassava_leaf/train_images/'):
    
    from catalyst.data.sampler import BalanceClassSampler
    
    #print("df", df)
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)
    #print("train_", train_)
        
    train_ds = CassavaDataset(train_, data_root, transforms=get_train_transforms(), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)
    valid_ds = CassavaDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True, one_hot_label=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=CFG['num_workers'],
        #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader




class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1])
        self.y  = self.y.to(device_all)#.cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss
    
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)
    
class MyCrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()
        #self.gamma = gamma

    def forward(self, y_pred, y_true):
        
        log_prob = nn.functional.log_softmax(y_pred, dim = 1)
        #print("log_prob shape is ",log_prob.shape)
        # 用之前得到的smoothed_labels来调整log_prob中每个值
        loss = - torch.sum(log_prob * y_true) / y_true.shape[0]
        return loss
    
class MyFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2):
        super(MyFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        #y_pred = torch.sigmoid(y_pred)
        y_pred = F.softmax(y_pred, dim=1)
        
        
        N = y_pred.size(0)
        C = y_pred.size(1)
        #print("y_true", y_true)

        class_mask = y_true.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = y_true.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print("y_true", class_mask)
        
        
        
        pt = y_pred * class_mask + (1-y_pred) * (1 - class_mask) #(1 - y_pred) * (1 - y_true)
        pt = torch.clamp(pt, epsilon, 1 - epsilon)
        CE = -torch.log(pt)
        FL = torch.pow(1 - pt, self.gamma) * CE
        loss = torch.sum(FL, dim=1)
        loss = torch.mean(loss, dim=0)
        return loss

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        #print("use fl...")

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        #print("P", P)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        #print("(P*class_mask",  P*class_mask)

        probs = (P*class_mask).sum(1).view(-1,1)
        #print("probs ", probs)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
def train_valid( model_save_path=None, label_smoothing=0):
    
    device = torch.device(CFG['device'])
    #criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    criterion_my = LabelSmoothingCrossEntropy()
    #criterion = MyFocalLoss()
    #criterion = FocalLoss(5)
    criterion = FocalCosineLoss()
    
    criterion = TaylorCrossEntropyLoss(n=2, smoothing=0.2) #log_0120
    # Specified GPU
    pathseresxt50 = "torch-class/hidden-dangers/models/se_resnext50_32x4d-a260b3a4.pth"
    model_name_c = 'se_resnext50_32x4d'
    #model_name_c = 'resnext50'
    #pathresxt50 = './torch-class/hidden-dangers/models/resnext50_32x4d-7cdf4587.pth'
    #model_name_c = 'efficient'

    
    base_path = "../data_cassava_leaf/"
    train = pd.read_csv(base_path + 'train.csv')
    train.head()

    seed_everything(CFG['seed'])
    
    ckpt_kfolds = "ckpt_kfolds_ourself_newloss_0203/"
    
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
    print("folds ", folds)
    
    all_fold_max_acc = [0,0,0,0,0]
    for fold, (trn_idx, val_idx) in enumerate(folds):
        #print("fold is ",fold, trn_idx[:], val_idx[:])

        mymodel = model.MyModel_all(num_class = 5,  model_name = model_name_c,  path = pathseresxt50)
        
        #if torch.cuda.device_count() > 1:
        #    mymodel = nn.DataParallel(mymodel, device_ids=[2])
        #    mymodel.to(device)
        mymodel.to(device)
        #mymodel.train()

        plist = [{'params': mymodel.parameters(), 'lr': 1e-3}]
        optimizer = optim.SGD(mymodel.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.00001, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 9 ], gamma=0.5)
        
        #optimizer = torch.optim.Adam(mymodel.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)

        
        all_acc = []
        val_loss_history = []
        val_loss_count = 0

        print('Training with {} started'.format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, data_root = base_path + '/train_images/')
        
        for epoch in range(CFG['epochs']):
            # every epoch should set train  
            mymodel.train()
            tr_loss = 0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            
            for step, (imgs, image_labels) in pbar:
                imgs = imgs.to(device).float()
                
                #print("imgs.shape", imgs.shape) 
                # *************long is int do cutmix dataset should float**************
                image_labels = image_labels.to(device).long() 

                image_preds = mymodel(imgs)   #output = model(input)
                loss = criterion(image_preds,  image_labels.squeeze())
                loss  = loss * 10
                loss.backward()
                tr_loss += loss.item()

                optimizer.step()
                optimizer.zero_grad()

                if step%print_freq*5 == 0 and step > 0:
                    print('loss', tr_loss/(step+1)*1.00,  'lr ',  optimizer.state_dict()['param_groups'][0]['lr'])

            
            #------valiate--------
            scheduler.step()
            
            print("eval ... ")
            mymodel.eval()
            image_preds_all_tmp = []
            image_targets_all = []
            loss_sum = 0
            sample_num = 0
            
            pbar_val = tqdm(enumerate(val_loader), total=len(val_loader))
            with torch.no_grad():
                for step, (imgs, image_labels) in pbar_val:

                    imgs = imgs.to(device).float()
                    image_labels = image_labels.to(device).long()

                    image_preds_val = mymodel(imgs)   #output = model(input)
                    #print(image_preds_val.shape)#, exam_pred.shape)
                    image_preds_all_tmp += [torch.argmax(image_preds_val, 1).detach().cpu().numpy()]
                    
                    if len(image_labels.shape) > 1 and image_labels.shape[1] > 1:
                        #if :
                        image_targets_all += [torch.argmax(image_labels, 1).detach().cpu().numpy()]
                    else:
                        image_targets_all += [image_labels.detach().cpu().numpy()]

                    loss = criterion(image_preds_val, image_labels)

                    loss_sum += loss.item()*image_labels.shape[0]
                    sample_num += image_labels.shape[0]  

                    if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
                        description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
                        pbar.set_description(description)

                image_preds_all = np.concatenate(image_preds_all_tmp)
                image_targets_all = np.concatenate(image_targets_all)
                print('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))

                acc = (image_preds_all==image_targets_all).mean()  

                #--------save pth---------
                model_save_path = ckpt_kfolds + 'best_' + CFG['model_arch'] + str(fold).zfill(4) 

            if len(val_loss_history)>0:# != []:
                    if val_loss_history[-1] < acc :
                        print('saving model...', fold, epoch, model_save_path)
                        torch.save(mymodel.state_dict(), model_save_path)
                        val_loss_history.append( acc )
                        val_loss_count = 0

                        all_fold_max_acc[fold] = acc
                    else:
                        print('valid_loss did not improve')
                        val_loss_count += 1

            else:
                val_loss_history.append(acc)
                print('saving model...')
                torch.save(mymodel.state_dict(), model_save_path)


            torch.save(mymodel.state_dict(),ckpt_kfolds + '{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch))
            
            
        del mymodel, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()
    print("max acc is ", all_fold_max_acc, sum(all_fold_max_acc)/5.0)
    
        

print_freq = 700
# train or test
VERSION = 'base_five_1.2'
all_root = "ckpt/"
save_path = all_root + 'ckpt_five/'
model_save_path = save_path + VERSION + '-class5_efficientb4_aug_new_fcl.pth'

mode = sys.argv[1]
if mode == 'train':
    train_valid(model_save_path)
