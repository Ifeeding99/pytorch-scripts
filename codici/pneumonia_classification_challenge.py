import torch
import matplotlib.pyplot as plt
import pydicom
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from torchmetrics.classification import BinaryAccuracy
import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True' #to visualize with plt.imshow()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_images_path = 'C:/Users/flavi/Downloads/rsna-pneumonia-detection-challenge/stage_2_train_images/'
label_path = "C:/Users/flavi/Downloads/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"


'''
prova pydicom

im = pydicom.read_file("C:/Users/flavi/Downloads/rsna-pneumonia-detection-challenge/stage_2_train_images/ff34eb2e-60eb-401a-81bc-8083dab28957.dcm").pixel_array
print(im)
plt.imshow(im)
plt.show()
'''

class PneumoniaDataset(Dataset):
    def __init__(self, transform=None):
        self.im_path = train_images_path
        self.list_images = os.listdir(self.im_path)
        self.target = pd.read_csv(label_path)
        self.target['patientId'] = self.target['patientId'].map(lambda x: x+'.dcm')
        self.l = len(self.list_images)
        self.transform = transform

    def __getitem__(self, idx):
        img = self.list_images[idx]
        df = self.target.loc[self.target['patientId']==img]
        label = df.iloc[0,5]
        img = pydicom.read_file(os.path.join(self.im_path,img)).pixel_array
        if self.transform:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return self.l

t = Compose([
    ToPILImage(),
    Resize((256,256)),
    ToTensor()
])

dataset = PneumoniaDataset(t)
#print(len(dataset)) # 26684
train_dataset, val_dataset = torch.utils.data.random_split(dataset,[21347,5337])
b_size = 64
train_loader = DataLoader(train_dataset, batch_size=b_size)
val_loader = DataLoader(val_dataset, batch_size=b_size)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=(1,1), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.conv_layer(x)
        x = self.lin(x)
        return x

model = Model()
model = model.to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
criterion = nn.BCELoss()
acc = BinaryAccuracy()
acc = acc.to(device)
n_epochs = 10

for epoch in range(n_epochs):
    loop = tqdm.tqdm(train_loader)
    for idx,(X,y) in enumerate(loop):
        X = X.to(device)
        y = y.to(device)
        y = torch.unsqueeze(y,1)
        y = y.float()
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc(pred,y)

        loop.set_description(f"epoch {epoch+1}/{n_epochs}")
        loop.set_postfix(loss=loss.item(), accuracy=acc.compute().item())

