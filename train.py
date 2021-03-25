from model import *
from data import *
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torchvision.models as models
import math
# from torch.nn import init
from torchvision import transforms
import time
from tfs import *
from mixup import *

NUM_CLASS = 5
BATCH_SIZE = 40
EPOCH = 30
RESUME = False 
RESUME_EPOACH = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



transform_train,transform_test = get_cub_transform()

dataset_train = DataSetsLeaf("/home/dourenyin/DATA/Kaggle-Center/datasets/2020","/home/dourenyin/DATA/Kaggle-Center/datasets/train.txt",transform_train)
dataset_test  = DataSetsLeaf("/home/dourenyin/DATA/Kaggle-Center/datasets/2020","/home/dourenyin/DATA/Kaggle-Center/datasets/test.txt",transform_test)

train_loader = DataLoader(dataset_train,   batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
validate_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

model = ModelResnest50()

if RESUME == True:
    ckpt_path = "checkpoints/resnest50_" + str(RESUME_EPOACH).zfill(4) + ".pth"
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model'])
model.cuda()

criterion = nn.CrossEntropyLoss(reduction='none').cuda()

params = [{'params': model.conv.parameters(), 'lr': 0.0005}, \
        {'params': model.classifier.parameters(), 'lr': 0.005} ]

optimizer = optim.SGD(params,0.0005,momentum=0.9,weight_decay=1e-4,nesterov=True)
model = nn.DataParallel(model)
BATCH = len(train_loader)
m = len(validate_loader)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1, last_epoch=-1)
best_acc = 0.0
best_epoach = 0
wmodel = None
for epoch in range(RESUME_EPOACH,EPOCH):
    model.train()
    EPOCH_START = time.time()
    print("Epoch #{0}/{1}:".format(epoch,EPOCH))
    train_loss = 0.0
    if wmodel is None:
        wmodel = model
    for batch,(x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.cuda()
        y = y.cuda()
        x,target_a,target_b,lam_a,lam_b = snapmix(x,y,wmodel)
        output,_ = model(x)
        loss_a = criterion(output, target_a)
        loss_b = criterion(output, target_b)
        loss = torch.mean(loss_a* lam_a + loss_b* lam_b)
        train_loss += loss
        print("Batch #{0}/{1}: ".format(batch+1,BATCH) + "Loss = %.6f"%float(loss))
        loss.backward()
        optimizer.step()

    model_save_path = "checkpoints/resnest_" + str(epoch).zfill(4) +  ".pth"
    # , 'optimizer':optimizer.state_dict(),
    state_model = {'model':model.state_dict(), 'epoch':epoch}
    torch.save(state_model, model_save_path)    
    scheduler.step()
    print("train_loss is" , train_loss/BATCH)
    model.eval()
    all_target = None
    all_pred = None

    valid_loss = 0.0
    for batch_v, (x_v, y_v) in enumerate(validate_loader):
        x_v = x_v.cuda()
        if all_target is None:
            all_target = y_v
        else:
            all_target = torch.cat((all_target, y_v), dim=0)

        with torch.no_grad():
            pred_t,_ = model(x_v)
            loss = torch.mean(criterion(pred_t,y_v.cuda()))
            valid_loss += loss
            if all_pred is None:
                all_pred = pred_t
            else:
                all_pred = torch.cat((all_pred, pred_t), dim=0)
    print("valid_loss",valid_loss/m)

    all_res = torch.argmax(all_pred,dim=1)
    acc = torch.sum(all_res.cpu() == all_target.cpu() )/all_target.shape[0]
    print(epoch, " acc: ", acc.item())
    if acc.item() > best_acc :
        best_acc = acc.item()
        best_epoach = epoch

        #统计每一类的精度
    for class_index in range(0,NUM_CLASS):
        class_gt_index = (all_target.cpu() == class_index)
        total = class_gt_index.sum().item()
        pred_tp = (all_res[class_gt_index].cpu() == class_index).sum().item()
        print("class ", str(class_index)," acc is ", pred_tp,total,pred_tp*1.0/total)
print(best_epoach,best_acc)