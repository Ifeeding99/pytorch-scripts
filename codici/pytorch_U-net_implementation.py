import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import tqdm
from torchmetrics.classification import BinaryAccuracy
from PIL import Image
import os
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, ElasticTransform, Compose, ColorJitter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 12

train_mask_csv = pd.read_csv("C:/Users/flavi/Downloads/carvana-image-masking-challenge/train_masks.csv/train_masks.csv")
train_path = "C:/Users/flavi/Downloads/carvana-image-masking-challenge/train"
train_mask_path = "C:/Users/flavi/Downloads/carvana-image-masking-challenge/train_masks/train_masks"

class MyDataset(Dataset):
    def __init__(self, transforms = None):
        self.train_path = train_path
        self.train_mask_path = train_mask_path
        self.list_images = os.listdir(self.train_path)
        self.list_masks = os.listdir(self.train_mask_path)
        self.transforms = transforms

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.train_path, self.list_images[item]))
        mask = Image.open(os.path.join(self.train_mask_path, self.list_masks[item]))
        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image,mask

    def __len__(self):
        return len(self.list_images)

t = Compose([
    Resize((256,256)),
    ToTensor()
])
dataset = MyDataset(t) # lenght = 5088
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [4070, 1018])
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

class U_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.first_conv_block = nn.Sequential(

             nn.Conv2d(in_channels = 3, out_channels=64, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),
             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),
             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),

        )
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))

        self.second_conv_block = nn.Sequential(

             nn.Conv2d(in_channels = 64, out_channels=128, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),
             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),
             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),

        )
        self.third_conv_block = nn.Sequential(

             nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),
             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),
             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),

        )

        self.fourth_conv_block = nn.Sequential(

             nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),
             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),
             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
             nn.ReLU(),

        )
        self.conv_5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), stride = (1,1), padding = (1,1))
        self.conv_6 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), stride = (1,1), padding = (1,1))
        self.conv_7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), stride = (1,1), padding = (1,1))

        self.upconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2,2), stride = (2,2))
        self.up_1_block = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )
        self.upconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        self.up_2_block = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        self.upconv_3 = nn.ConvTranspose2d(in_channels=256,out_channels=128, kernel_size=(2,2),stride=(2,2))
        self.up_3_block = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        self.final_upconv = nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=(2,2),stride=(2,2))
        self.final_block = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=(1, 1)),
            nn.Sigmoid()
        )


    def forward(self, x):
        x_1 = self.first_conv_block(x)
        x_1_pool = self.pool(x_1)
        x_2 = self.second_conv_block(x_1_pool)
        x_2_pool = self.pool(x_2)
        x_3 = self.third_conv_block(x_2_pool)
        x_3_pool = self.pool(x_3)
        x_4 = self.fourth_conv_block(x_3_pool)
        x_4_pool = self.pool(x_4)
        x_5 = self.conv_5(x_4_pool)
        x_5_relu = self.relu(x_5)
        x_6 = self.conv_6(x_5_relu)
        x_6_relu = self.relu(x_6)
        x_7 = self.conv_7(x_6_relu)
        x_7_relu = self.relu(x_7)

        # ora la risalita

        x_7_r = self.upconv_1(x_7_relu)
        x_8 = torch.cat((x_4,x_7_r), axis = 1) #axis = 1 because you have [batch,C,H,W]
        x_8 = self.up_1_block(x_8)

        x_8_r = self.upconv_2(x_8)
        x_9 = torch.cat((x_3, x_8_r), axis = 1)
        x_9 = self.up_2_block(x_9)

        x_9_r = self.upconv_3(x_9)
        x_10 = torch.cat((x_2,x_9_r), axis = 1)
        x_10 = self.up_3_block(x_10)

        x_10_r = self.final_upconv(x_10)
        x_11 = torch.cat((x_1,x_10_r), axis = 1)
        output = self.final_block(x_11)

        return output


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs*targets).sum()
        dice = (2. * intersection + 1)/(inputs.sum() + targets.sum() + 1)
        bce = nn.BCELoss()
        return bce(inputs,targets)+dice


NN = U_net()
NN = NN.to(device)
n_epochs = 100
optimizer = torch.optim.SGD(NN.parameters(), lr = 1e-3, nesterov = True, momentum = 0.9)
loss_fn = DiceLoss()
print('Training start \n')

for epoch in range(n_epochs):
    train_loop = tqdm.tqdm(train_loader)
    for batch,(x,y) in enumerate(train_loop):
        x = x.to(device)
        y = y.to(device)
        pred = NN(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loop.set_description(f'epoch {epoch+1}/{n_epochs}')
        train_loop.set_postfix(loss = loss.item())


