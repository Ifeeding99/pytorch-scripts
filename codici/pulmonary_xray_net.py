import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
from torchvision.transforms import ToTensor, Resize, Compose, InterpolationMode, CenterCrop
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.tuner import Tuner
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import StochasticWeightAveraging, EarlyStopping


torch.set_float32_matmul_precision('medium')
class PneumoniaDataset(Dataset):
    def __init__(self, t=None):
        super().__init__()
        self.t = t
        self.images_path = 'C:/Users/flavi/Downloads/CXR8/CXR8/images_extracted'
        self.labels_path = 'C:/Users/flavi/Downloads/CXR8/CXR8/data_labels.csv'
        self.list_images = os.listdir(self.images_path)
        self.labels = pd.read_csv(self.labels_path)
        self.l = len(self.list_images)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.images_path, self.list_images[item]))
        img = ImageOps.grayscale(img)
        df_row = self.labels[self.labels['id'] == self.list_images[item]]
        Atelectasis = int(df_row.iloc[0,1])
        Cardiomegaly = int(df_row.iloc[0,2])
        Effusion = int(df_row.iloc[0,3])
        Infiltration = int(df_row.iloc[0,4])
        Mass = int(df_row.iloc[0,5])
        Nodule = int(df_row.iloc[0,6])
        Pneumonia = int(df_row.iloc[0,7])
        Pneumothorax = int(df_row.iloc[0,8])
        Consolidation = int(df_row.iloc[0,9])
        Edema = int(df_row.iloc[0,10])
        Emphysema = int(df_row.iloc[0,11])
        Fibrosis = int(df_row.iloc[0,12])
        Pleural_Thickening = int(df_row.iloc[0,13])
        Hernia = int(df_row.iloc[0,14])
        No_Finding = int(df_row.iloc[0,15])
        if self.t is not None:
            img = self.t(img)
        tensor_label = torch.tensor([Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,
                                        Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
                                        Pleural_Thickening, Hernia, No_Finding], dtype=torch.float32)
        return img,tensor_label

    def __len__(self):
        return self.l


class PneumoniaData(L.LightningDataModule):
    def __init__(self, t, batch_size):
        super().__init__()
        self.t = t
        self.batch_size = batch_size

    def setup(self,stage):
        self.p = PneumoniaDataset(self.t)
        self.val_split = 0.8
        self.train_examples = round(len(self.p)*0.8)
        self.val_examples = len(self.p) - self.train_examples
        self.train_dataset, self.val_dataset = random_split(self.p,[self.train_examples, self.val_examples])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size, num_workers=10)

    def val_dataloader(self) :
        return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=10)

img_size = 256
t = Compose([
    Resize((img_size,img_size),interpolation=InterpolationMode.BILINEAR),
    CenterCrop((224,224)),
    ToTensor()
])

class TransferLearning(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.acc = BinaryAccuracy()
        self.lr = lr
        self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.conv1 = nn.Conv2d(1,64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias = False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.GELU(),
            nn.Linear(in_features=1024, out_features=64),
            nn.GELU(),
            nn.Linear(in_features=64,out_features=15)
        )
    def forward(self, x):
        return self.resnet(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),lr=self.lr,momentum=0.9, nesterov=True)

    def training_step(self, batch, batch_idx):
        X,y = batch
        pred = self.forward(X)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        acc = self.acc(pred,y)
        self.log_dict({'train_loss':loss, 'accuracy':acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X,y = batch
        pred = self.forward(X)
        loss = F.binary_cross_entropy_with_logits(pred,y)
        acc = self.acc(pred,y)
        self.log_dict({'val_loss':loss, 'accuracy':acc}, prog_bar=True)
        return loss


if __name__ == '__main__':
    swa = StochasticWeightAveraging(0.05)
    stop = EarlyStopping(monitor = 'val_loss', min_delta=0.005, patience=3, mode='min')
    data = PneumoniaData(t, 64)
    model = TransferLearning(1e-3)
    trainer = Trainer(accelerator='gpu', precision='16-mixed', max_epochs=30, gradient_clip_val=1,
                      gradient_clip_algorithm='value', callbacks=[swa,stop])
    tuner = Tuner(trainer)
    #lr_finder = tuner.lr_find(model, data)
    #fig = lr_finder.plot(suggest=True)
    #plt.show()
    #tuner.scale_batch_size(model,data)
    trainer.fit(model,data)
