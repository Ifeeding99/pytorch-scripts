import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torchmetrics.classification import BinaryAccuracy
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, ToTensor, RandomCrop, Resize
from PIL import Image
from torch.utils.data import Dataset, DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DenseBlock, self).__init__()
        self.single_conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch*4, kernel_size=(1,1), stride=(1,1))
        self.conv1 = nn.Conv2d(in_channels=out_ch*4, out_channels=out_ch, kernel_size=(3,3), padding=(1,1),
                               stride=(1,1))
        nn.init.kaiming_normal_(self.conv1.weight)

        self.single_conv2 = nn.Conv2d(in_channels=out_ch+out_ch*4, out_channels=out_ch*4, kernel_size=(1, 1), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=out_ch*4, out_channels=out_ch, kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1))
        nn.init.kaiming_normal_(self.conv1.weight)

        self.single_conv3 = nn.Conv2d(in_channels=2*out_ch+4*out_ch, out_channels=out_ch*4, kernel_size=(1, 1), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=out_ch*4, out_channels=out_ch, kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1))
        nn.init.kaiming_normal_(self.conv1.weight)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self,x):
        out1 = self.single_conv1(x)
        to_cat1 = out1
        out1 = self.conv1(out1)
        out1 = self.relu(out1)
        out1 = self.bn(out1)
        x1 = torch.cat([to_cat1,out1], dim=1)
        out2 = self.single_conv2(x1)
        to_cat2 = out2
        out2 = self.conv2(out2)
        out2 = self.relu(out2)
        out2 = self.bn(out2)
        x2 = torch.cat([x1,out2], dim = 1)
        out3 = self.single_conv3(x2)
        out3 = self.conv3(out3)
        out3 = self.relu(out3)
        out3 = self.bn(out3)
        return out3

class TransitionLayer(nn.Module):
    def __init__(self, in_ch, theta):
        super(TransitionLayer, self).__init__()
        out_ch = int(theta * in_ch)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=int(theta*in_ch), kernel_size=(1,1), stride=(2,2), padding=(1,1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(int(theta*in_ch))

    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(7,7), padding=(3,3), stride=(2,2))
        self.d1 = DenseBlock(in_ch=12, out_ch=24)
        self.t1 = TransitionLayer(in_ch=24, theta=0.5)
        self.d2 = DenseBlock(in_ch=12, out_ch=24)
        self.t2 = TransitionLayer(in_ch=24, theta=0.5)
        self.d3 = DenseBlock(in_ch=12, out_ch=12)
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features=12*30*30, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.d1(x)
        x = self.t1(x)
        x = self.d2(x)
        x = self.t2(x)
        x = self.d3(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


class CatsDogsDataset (Dataset):
    def __init__(self, t):
        self.image_path = 'C:/Users/flavi/Downloads/dogs-vs-cats/train'
        self.list_images = os.listdir(self.image_path)
        self.l = len(self.list_images)
        self.t = t

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.image_path,self.list_images[i]))
        label = 1 if 'dog' in self.list_images[i] else 0
        if self.t:
            image = self.t(image)
        return image,label

    def __len__(self):
        return self.l

t = Compose(
[    Resize((256,256)),
    RandomCrop((224,224)),
    RandomRotation(45),
    RandomVerticalFlip(),
    RandomHorizontalFlip(),
    ToTensor()]
)
train_dataset, val_dataset = torch.utils.data.random_split(CatsDogsDataset(t), [20000,5000])
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=10, shuffle=True)

model = DenseNet()
model.to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, nesterov = True, momentum=0.9)
n_epochs = 10
criterion = torch.nn.BCEWithLogitsLoss()
acc_fn = BinaryAccuracy().to(device)
if __name__ == '__main__':

    for epoch in range(n_epochs):
        loop = tqdm(train_loader)
        for i,batch in enumerate(loop):
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            y = y.view(-1,1)
            pred = model(x)
            loss = criterion(pred,y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_fn(pred,y)

            loop.set_description(f'epoch{epoch+1}/{n_epochs}')
            loop.set_postfix(loss=loss.item(), acc = acc_fn.compute().item())

torch.save(model, 'C:/Users/flavi/anaconda3/envs/pytorch_env/codici/shitty_densenet.pth')

