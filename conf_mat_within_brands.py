import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.transforms import transforms
import numpy as np 
from torch.autograd import Variable
from PIL import Image
import json 
import os
from matplotlib import pyplot as plt

input_dim = 112
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./models/fgc_bilinear.model"
checkpoint = torch.load(model_path)

with open("hyper_params.json") as hp:
    data = json.load(hp)
    num_classes = data["num_classes"]
    root_dir = data["root_directory"]
    batch_size = data["batch_size"]
    num_workers = data["num_workers"]

model_ft = models.vgg19(pretrained=True)
class vgg19_see_smart(nn.Module):
    def __init__(self, originalModel):
        super(vgg19_see_smart, self).__init__()
        self.features = models.vgg19(pretrained=True).features
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
        x = x.view(N, 512, 7*7)
        x = torch.bmm(x, torch.transpose(x,1,2))/ (7**2) # Bilinear
        x = x.view(N, 512**2)
        x = torch.sqrt(x + 1e-12)
        x = nn.functional.normalize(x)
        x = self.fc(x)
        #x = x.view()
        return x
model_ft = vgg19_see_smart(model_ft)
model_ft.load_state_dict(checkpoint)
model_ft.to(device) 

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


image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x),
                                          data_transforms[x])
            for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
            for x in ['train', 'val']}

# Within brands conf

from sklearn.metrics import confusion_matrix

predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

def brand_mod(x_list):
    new_list = []
    for i in x_list:
        if i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,44,45,48,49,54,55,56,57,58]:
            new_list.append(1)
        elif i in [16,17,42,43,61]:
            new_list.append(2)
        elif i in [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,50,51,52,53,59]:
            new_list.append(3)
        else:
            new_list.append(4)
    return new_list

with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)

        # Append batch prediction results
        predlist=torch.cat([predlist,preds.view(-1).cpu()])
        lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

    pred_list = brand_mod(predlist.numpy())
    lbl_list = brand_mod(lbllist.numpy())
    print(pred_list)
    print(lbl_list)

# Confusion matrix
conf_mat=confusion_matrix(lbl_list, pred_list)
print(conf_mat)

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print(class_accuracy)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + [i for i in range(4)])
ax.set_yticklabels([''] + [i for i in range(4)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()