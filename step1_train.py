import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import torch.nn.functional as F
import timm
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

torch.cuda.empty_cache() 

n_epoch =  100
s_batch = 8

data_dir = './dataset/'
'''
transforms.CenterCrop(224),
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomRotation(10),
transforms.ColorJitter(brightness=0.2, contrast=0.2),
transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
'''
'''
transforms.CenterCrop(224),
'''
data_transforms = {
    'train': transforms.Compose([

        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=s_batch, shuffle=True, num_workers=0)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
test_acc = []

print(dset_sizes)
dset_classes = dsets['train'].classes
n_class = len(dset_classes)
print(dset_classes,n_class)

use_gpu = torch.cuda.is_available()
print(use_gpu)

inputs, classes = next(iter(dset_loaders['train']))
out = torchvision.utils.make_grid(inputs)

def train_model(model, criterion, optimizer, lr_scheduler, num_epoch=10, csv_path="train_log.csv"):
    since = time.time()

    best_model = model
    best_acc = 0.0
    best_f1 = 0.0
    best_loss = float("inf")

    acc_records = {'train': [], 'val': []}
    f1_records = {'train': [], 'val': []}

    # ⚠️ 先清空舊檔案，並寫上標題
    with open(csv_path, "w") as f:
        f.write("epoch,phase,loss,accuracy,f1\n")

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            epoch_preds_all = []
            epoch_labels_all = []

            for data in dset_loaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)

                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                epoch_preds_all.append(preds.detach().cpu())
                epoch_labels_all.append(labels.detach().cpu())

            # ---- 計算指標 ----
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            y_pred = torch.cat(epoch_preds_all).numpy()
            y_true = torch.cat(epoch_labels_all).numpy()
            epoch_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_f1))

            acc_records[phase].append(epoch_acc)
            f1_records[phase].append(epoch_f1)

            # ---- 只在 val 階段更新 best model ----
            if phase == 'val' and epoch_acc > best_acc: 
                best_acc = epoch_acc 
                best_model = copy.deepcopy(model)

            # ---- 每個 epoch 寫入 CSV ----
            with open(csv_path, "a") as f:
                f.write(f"{epoch},{phase},{epoch_loss:.4f},{epoch_acc:.4f},{epoch_f1:.4f}\n")

            if phase == 'val':
                cm = confusion_matrix(y_true, y_pred)
                print(cm)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))
    return best_model, best_acc, acc_records, f1_records



def exp_lr_scheduler(optimizer, epoch, init_lr=1e-3, lr_decay_epoch= 80 ): # 18 12 9  0.01
    lr = init_lr * (0.5 ** (epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print("LR is set to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# Finetuning the convnet
model_select = 'resnet50' #vit_base_patch16_224

model_ft = timm.create_model(model_select, pretrained=True)
#model_ft = timm.create_model(model_select, pretrained=True,img_size=512) #'vit_base_patch16_224'
'''
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, n_class)

'''
in_features = model_ft.get_classifier().in_features
model_ft.reset_classifier(num_classes=n_class)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.009, momentum=0.9)

model_ft, val_acc, acc_records, f1_records = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    num_epoch=n_epoch,
    csv_path= model_select +"_train_val_log_"+str(n_epoch)+"_meg.csv"   # 這裡會逐 epoch append
)

acc_str = str(val_acc).split('.')[1]
torch.save(model_ft,model_select+'_ep_'+str(n_epoch)+'.pkl')

df = pd.DataFrame(acc_records)
df.to_csv(model_select+'_Records_ep_'+str(n_epoch)+'_meg.csv', index=False)
