import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tqdm
from collections import OrderedDict
from torchvision.transforms import ToTensor, Resize, Compose
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, random_split
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.tuner import Tuner

img_size = 224
b_size = 32
path = 'C:/Users/flavi/Downloads/dogs-vs-cats/train'
torch.set_float32_matmul_precision('medium')

class MyDataset(Dataset):
    def __init__(self, t=None):
        super().__init__()
        self.path = path
        self.list_images = os.listdir(path)
        self.l = len(self.list_images)
        self.t = t

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.path, self.list_images[item]))
        label = 0 if 'cat' in self.list_images[item] else 1
        if self.t:
            img = self.t(img)
        return img, label

    def __len__(self):
        return self.l

t = Compose([
    Resize((img_size,img_size)),
    ToTensor()
])

class DataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers, train_split):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split

    def setup(self, stage):
        self.my_dataset = MyDataset(t)
        self.n_train_examples = round(self.train_split * len(self.my_dataset))
        self.n_val_examples = len(self.my_dataset) - self.n_train_examples
        self.train_dataset, self.val_dataset = random_split(self.my_dataset,[self.n_train_examples, self.n_val_examples])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(3,3), stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size = (3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.layers(x)
        return x+x1

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,1), stride=(2,2))
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(3,3), stride=(2,2), padding = (1,1)),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,3), stride = (1,1), padding=(1,1)),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x2 = self.conv_1(x)
        x1 = self.layers(x)
        return x2+x1

class MyResNet(L.LightningModule):
    def __init__(self, lr=1e-3, batch_size=32):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.acc = BinaryAccuracy()
        self.first_conv = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(7,7),stride=(2,2), padding = (3,3))
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.block1 = BasicBlock(in_channels=64, out_channels=64)
        self.block2 = BasicBlock(in_channels=64, out_channels=64)
        self.block3 = BasicBlock(in_channels=64, out_channels=64)
        self.block4 = BottleneckBlock(in_channels=64,out_channels=128)
        self.block5 = BasicBlock(in_channels=128, out_channels=128)
        self.block6 = BasicBlock(in_channels=128, out_channels=128)
        self.block7 = BasicBlock(in_channels=128, out_channels=128)
        self.block8 = BottleneckBlock(in_channels=128, out_channels=256)
        self.block9 = BasicBlock(in_channels=256, out_channels=256)
        self.block10 = BasicBlock(in_channels=256, out_channels=256)
        self.block11= BasicBlock(in_channels=256, out_channels=256)
        self.block12 = BasicBlock(in_channels=256, out_channels=256)
        self.block13 = BasicBlock(in_channels=256, out_channels=256)
        self.block14 = BottleneckBlock(in_channels=256, out_channels=512)
        self.block15 = BasicBlock(in_channels=512, out_channels=512)
        self.block16 = BasicBlock(in_channels=512, out_channels=512)
        self.avg_pool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_features=512, out_features=1000)
        self.output = nn.Linear(in_features=1000, out_features=1)


    def forward(self, x):
        x = self.first_conv(x)
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.avg_pool(x)
        x = self.flat(x)
        x = F.relu(self.fc(x))
        x = self.output(x)
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

    def training_step(self, batch, batch_idx):
        x,y = batch
        y = y.view(-1,1)
        y = y.float()
        pred = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        acc = self.acc(pred, y)
        self.log_dict({'train_loss':loss, 'train_acc':acc}, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y = y.view(-1,1)
        y = y.float()
        pred = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(pred,y)
        acc = self.acc(pred,y)
        self.log_dict({'val_loss':loss, 'val_acc':acc}, prog_bar=True,on_step=True)
        return loss

if __name__ == '__main__':
    dModule = DataModule(batch_size=256, num_workers=10, train_split=0.8)
    model = MyResNet()
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.008, patience=5)
    swa = StochasticWeightAveraging(1e-2)
    trainer = L.Trainer(accelerator='gpu', precision='16-mixed', max_epochs=100, callbacks=[early_stop, swa], accumulate_grad_batches=5,
                        gradient_clip_val=1, gradient_clip_algorithm='value')
    #tuner = Tuner(trainer)
    #lr_finder = tuner.lr_find(model,train_loader, val_loader)
    #print(lr_finder.results)
    #fig = lr_finder.plot(suggest=True)
    #fig.show()
    #plt.show()
    #tuner.scale_batch_size(model,dModule) 256
    trainer.fit(model, dModule)











