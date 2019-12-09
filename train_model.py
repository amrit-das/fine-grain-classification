from __future__ import print_function, division
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
from center_loss import CenterLoss
import time
import os
import copy
from cross_entropy import CrossEntropyLoss
import torch.nn.functional as F 

input_dim = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_dim),
        transforms.CenterCrop(input_dim),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_dim),
        transforms.CenterCrop(input_dim),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

with open("hyper_params.json") as hp:
    data = json.load(hp)
    root_dir = data["root_directory"]
    num_classes = data["num_classes"]
    num_epochs = data["num_epochs"]
    batch_size = data["batch_size"]
    num_workers = data["num_workers"]
    lr = data["learning_rate"]
    optim_name = data["optimizer"] 
    momentum = data["momentum"]
    step_size = data["step_size"]
    gamma = data["gamma"]


image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x),
                                          data_transforms[x])
            for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
            for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

class_map={}
for x in range (0,len(class_names)):
    class_map[x]=class_names[x]

with open('class_mapping.json', 'w') as outfile:  
    json.dump(class_map, outfile, indent = 4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)
#device = torch.device("cpu")

def save_models(epoc448hs, model):
    print()
    torch.save(model.state_dict(), "./models/trained_fgc.model")
    print("****----Checkpoint Saved----****")
    print()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('_' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer[0].zero_grad()
                optimizer[1].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss1 = criterion[0](outputs, labels)
                    loss2 = criterion[1](outputs, labels)
                    total_loss = loss1 + loss2

                    if phase == 'train':
                        total_loss.backward()
                        optimizer[0].step()
                        optimizer[1].step()
                        scheduler.step()

                running_loss += total_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train' and epoch_acc > best_acc:
                save_models(epoch,model)
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

model_ft = models.vgg19(pretrained=True)

class vgg19_see_smart(nn.Module):
    def __init__(self, originalModel):
        super(vgg19_see_smart, self).__init__()
        self.features = torchvision.models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(self.features.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.fc = nn.Linear(512*512, num_classes)
        nn.init.kaiming_normal_(self.fc.weight.data)
        
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data, val=0)
    
    def forward(self, x):
        N = x.size()[0]
        #print(N)
        x = self.features(x)
        x = x.view(N, 512, 14*14)
        x = torch.bmm(x, torch.transpose(x,1,2))/ (14**2) # Bilinear
        x = x.view(N, 512**2)
        x = torch.sqrt(x + 1e-12)
        x = nn.functional.normalize(x)
        x = self.fc(x)
        #x = x.view()
        return x

model_ft = vgg19_see_smart(model_ft)
model_ft = model_ft.to(device)

celoss = CrossEntropyLoss(smooth_eps=0.1).to(device)
centerloss = CenterLoss(num_classes=num_classes,feat_dim=507,use_gpu=True ).to(device)

criterion = [celoss,centerloss]

max_lr = 0.001
min_lr = 0.00001

one_cycle = 20
num_cycle = 3
max_epochs = int(num_classes*one_cycle)

net_optimizer = torch.optim.SGD(model_ft.parameters(), max_lr, momentum=0.9, weight_decay=1e-4)
cl_optimimzer = torch.optim.SGD(centerloss.parameters(), max_lr, momentum=0.9, weight_decay=1e-4)

optimizer = [net_optimizer, cl_optimimzer]
lr_scheduler = lr_scheduler.StepLR(net_optimizer, step_size=step_size, gamma=gamma)

model_ft = train_model(model_ft, criterion, optimizer, lr_scheduler,num_epochs=num_epochs)
