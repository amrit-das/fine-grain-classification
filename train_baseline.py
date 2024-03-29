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
import time
import os
import copy
import torch.nn.functional as F 

input_dim = 112 

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_dim),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
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

def save_models(epochs, model):
    print()
    torch.save(model.state_dict(), "./models/baseline.model")
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
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
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

for param in model_ft.parameters():
   param.requires_grad = False

num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

#model_ft.aux_logits = False
#num_ftrs = model_ft.fc.in_features

'''class vgg16_see_smart(nn.Module):
    def __init__(self, originalModel):
        super(vgg16_see_smart, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features = nn.Sequential(*list(self.features.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False
        
        #self.adaptive_pool = nn.AvgPool2d(2)
        #self.conv1 = nn.Conv2d(1000, 2000, 3)
        
        self.fc = nn.Linear(512*512, num_classes)
        
        #self.dense1 = nn.Linear(512,512)
        #self.dense2 = nn.Linear(512,62)


        #nn.init.kaiming_normal_(self.fc.weight.data)
        #nn.init.constant_(self.fc.bias.data, val=0)
    
    def forward(self, x):
        N = x.size()[0]
        print(N)
        x = self.features(x)
        x = x.view(N, 512, 28*28)
        x = torch.bmm(x, torch.transpose(x,1,2))/ (28**2) # Bilinear
        x = x.view(N, 512**2)
        x = torch.sqrt(x + 1e-5)
        x = nn.functional.normalize(x)
        x = self.fc(x)
        #x = x.view()
        return x'''

#model_ft = vgg16_see_smart(model_ft)

print(model_ft)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=num_epochs)

