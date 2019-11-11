import torch
from torchvision import models, datasets, transforms
from torchvision.transforms import transforms
import numpy as np 
from torch.autograd import Variable
from PIL import Image
import json 
import os

input_dim = 112
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models/trained_vgg.model"
checkpoint = torch.load(model_path)

with open("hyper_params.json") as hp:
    data = json.load(hp)
    num_classes = data["num_classess"]
    root_dir = data["root_directory"]
    batch_size = data["batch_size"]
    num_workers = data["num_workers"]

model = models.vgg19(pretrained=True)

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

confusion_matrix = torch.zeroes(num_classes, num_classes)
with torch.no_grad():
    for i, (inputs,classes) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t,p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
print(confusion_matrix)



