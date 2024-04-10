import torch
import PIL
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Resize
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
print(f"Using {device}")
dataset_path = 'C:/Users/flavi/Downloads/dogs-vs-cats/train'

class My_Dataset(Dataset):
    def __init__(self, transforms = None):
        self.path = dataset_path
        self.image_list = os.listdir(dataset_path)
        self.l = len(self.image_list)
        print()
        self.transforms = transforms

    def __len__(self):
        return self.l

    def __getitem__(self, item):
        im = PIL.Image.open(os.path.join(self.path, self.image_list[item]))
        if 'cat' in self.image_list[item]:
            label = torch.tensor([0], dtype= torch.float32)
        else:
            label = torch.tensor([1], dtype= torch.float32)

        if self.transforms:
            im = self.transforms(im)

        return (im,label)

t = Compose([
    Resize((256,256)),
    ToTensor()
])
dataset = My_Dataset(transforms = t)
train_dataset, val_dataset = torch.utils.data.random_split(dataset,[20000, 5000])
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle = True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, stride=(1,1), kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, stride = (1,1), kernel_size = (3,3), padding = (1,1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride = (1,1), padding=(1,1))
        self.d1 = nn.Linear(in_features=64*64*64, out_features=256)
        self.d2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = nn.Flatten()(x)
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.sigmoid(x)

        return x

net = CNN()
net = net.to(device)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 1e-3, nesterov = True, momentum = 0.9)
train_acc_f = BinaryAccuracy()
train_acc_f = train_acc_f.to(device)
val_acc_f = BinaryAccuracy()
val_acc_f = val_acc_f.to(device)
n_epochs = 100

print(f"train loader batches:  {len(train_loader)}")

data, targets = next(iter(train_loader))
train_acc = 0
val_acc = 0

for epoch in range(n_epochs):
    loop = tqdm.tqdm(train_loader)
    for batch,(x,y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)
        pred = net(x)

        optimizer.zero_grad()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        train_acc_f(pred, y)
        train_acc = train_acc_f.compute()
        train_acc_f.reset()

        loop.set_description(f"epoch {epoch+1}/{n_epochs}")
        loop.set_postfix(training_accuracy=train_acc.item(),training_loss=loss.item())


    val_loop = tqdm.tqdm(val_loader)
    for val_batch,(x_val,y_val) in enumerate(val_loop):
        with torch.no_grad():
            net.eval()
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            pred_val = net(x_val)
            val_loss = loss_fn(pred_val, y_val)

            val_acc_f(pred_val, y_val)
            val_acc = val_acc_f.compute()
            val_acc_f.reset()

            val_loop.set_description(f"validation at epoch {epoch + 1}/{n_epochs}")
            val_loop.set_postfix(validation_accuracy=val_acc.item(), validation_loss=val_loss.item())

    print("\n \n")








