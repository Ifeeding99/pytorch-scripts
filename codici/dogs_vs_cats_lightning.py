import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchmetrics.classification import BinaryAccuracy
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, Resize, ToTensor

images_path = 'C:/Users/flavi/Downloads/dogs-vs-cats/train'
img_size = 256
b_size = 32
torch.set_float32_matmul_precision('medium')

class MyDataset(Dataset):
    def __init__(self, t):
        super().__init__()
        self.t = t
        self.path = images_path
        self.list_images = os.listdir(images_path)
        self.l = len(self.list_images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.list_images[idx]))
        label = 0 if 'cat' in self.list_images[idx] else 1
        if self.t:
            img = self.t(img)
        return img,label

    def __len__(self):
        return self.l

t = Compose([
    Resize((img_size,img_size)),
    #RandomHorizontalFlip(p=0.5),
    #RandomVerticalFlip(p=0.5),
    #RandomRotation(45),
    ToTensor()
])

dataset = MyDataset(t)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths= [20000,5000])
train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=b_size, num_workers=10)

class NN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=128*61*61, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1)
        )
        self.acc = BinaryAccuracy()

    def forward(self, x):
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        x,y = batch
        y = torch.unsqueeze(y,dim=1)
        y = y.float()
        pred = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(pred,y)
        a = self.acc(pred,y)
        self.log_dict(dictionary={'train_loss':loss, 'accuracy':a},prog_bar=True,on_step=True,on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y = torch.unsqueeze(y,dim=1).float()
        pred = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(pred,y)
        a = self.acc(pred,y)
        self.log_dict(dictionary={'val_loss':loss, 'accuracy':a}, prog_bar=True, on_step=True, on_epoch=True)
        return loss



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == '__main__':
    model = NN()
    trainer = L.Trainer(accelerator='gpu', precision='16-mixed', max_epochs=30)
    trainer.fit(model, train_loader, val_loader)