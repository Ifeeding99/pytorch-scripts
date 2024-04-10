import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.transforms import Resize, ToTensor, RandomRotation, RandomVerticalFlip, RandomHorizontalFlip, Compose, CenterCrop, InterpolationMode
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner
from typing import Optional
import torch.nn as nn
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import StochasticWeightAveraging, EarlyStopping
from sklearn.metrics import cohen_kappa_score
from torchmetrics.classification import MulticlassCohenKappa

torch.set_float32_matmul_precision('medium')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_images_path = 'C:/Users/flavi/Downloads/aptos2019-blindness-detection/train_images'
train_csv_path = 'C:/Users/flavi/Downloads/aptos2019-blindness-detection/train.csv'

class RetinopathyDataset(Dataset):
    def __init__(self, images_path, label_path, t=None):
        super().__init__()
        self.t = t
        self.images_path = images_path
        self.label_path = label_path
        self.labels = pd.read_csv(self.label_path)
        self.images_list = os.listdir(self.images_path)
        self.l = len(self.images_list)

    def __getitem__(self, item):
        image = self.images_list[item][:-4]
        name = image
        label = self.labels[self.labels.id_code == image].iloc[0,1]
        image = Image.open(os.path.join(self.images_path,image + '.png'))
        if self.t:
            image = self.t(image)

        if image.isnan().any() or image.isinf().any():
            print(name)
        return image, label

    def __len__(self):
        return self.l

transforms = Compose([
    Resize((256,256), interpolation=InterpolationMode.BILINEAR),
    CenterCrop((224,224)),
    RandomRotation(45),
    RandomVerticalFlip(),
    RandomHorizontalFlip(),
    ToTensor()
])

class RetinopathyData(L.LightningDataModule):
    def __init__(self, train_images_path, train_images_csv, batch_size=32,t=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_images_path = train_images_path
        self.train_images_csv = train_images_csv
        self.t = t

    def setup(self, stage: str):
        dataset = RetinopathyDataset(self.train_images_path, self.train_images_csv, self.t)
        train_examples = int(0.8*len(dataset))
        val_examples = len(dataset) - train_examples
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset,[train_examples, val_examples])


    def train_dataloader(self) :
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=10, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=10)


class WeightedKappaLoss(nn.Module):
    """
    Implements Weighted Kappa Loss. Weighted Kappa Loss was introduced in the
    [Weighted kappa loss function for multi-class classification
      of ordinal data in deep learning]
      (https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666).
    Weighted Kappa is widely used in Ordinal Classification Problems. The loss
    value lies in $[-\infty, \log 2]$, where $\log 2$ means the random prediction
    Usage: loss_fn = WeightedKappaLoss(num_classes = NUM_CLASSES)
    """

    def __init__(
            self,
            num_classes: int,
            mode: Optional[str] = 'quadratic',
            name: Optional[str] = 'cohen_kappa_loss',
            epsilon: Optional[float] = 1e-10):

        """Creates a `WeightedKappaLoss` instance.
            Args:
              num_classes: Number of unique classes in your dataset.
              weightage: (Optional) Weighting to be considered for calculating
                kappa statistics. A valid value is one of
                ['linear', 'quadratic']. Defaults to 'quadratic'.
              name: (Optional) String name of the metric instance.
              epsilon: (Optional) increment to avoid log zero,
                so the loss will be $ \log(1 - k + \epsilon) $, where $ k $ lies
                in $ [-1, 1] $. Defaults to 1e-10.
            Raises:
              ValueError: If the value passed for `weightage` is invalid
                i.e. not any one of ['linear', 'quadratic']
            """

        super(WeightedKappaLoss, self).__init__()
        self.num_classes = num_classes
        if mode == 'quadratic':
            self.y_pow = 2
        if mode == 'linear':
            self.y_pow = 1

        self.epsilon = epsilon

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        y = torch.eye(num_classes).to(device)
        y_true = y[y_true]

        y_true = y_true.float()

        repeat_op = torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes)).to(device)
        repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        weights = repeat_op_sq / ((num_classes - 1) ** 2)

        pred_ = y_pred ** self.y_pow
        pred_norm = pred_ / (self.epsilon + torch.reshape(torch.sum(pred_, 1), [-1, 1]))

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T, y_true)

        bsize = y_pred.size(0)
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.matmul(torch.reshape(hist_rater_a, [num_classes, 1]),
                                      torch.reshape(hist_rater_b, [1, num_classes]))
        denom = torch.sum(weights * expected_probs / bsize)

        return nom / (denom + self.epsilon)

    def forward(self, y_pred, y_true):
        return self.kappa_loss(y_pred, y_true)

class MyNN(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.k = MulticlassCohenKappa(num_classes=5,weights='quadratic')
        self.criterion = torch.nn.MSELoss()
        self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=500),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=500, out_features=100),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=100, out_features=1),
        )
        for m in self.resnet.fc:
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, X):
        return self.resnet(X)

    def configure_optimizers(self):
        return torch.optim.SGD(params=self.parameters(), lr=self.lr, weight_decay=1e-6, nesterov=True, momentum=0.9)

    def training_step(self, batch, batch_idx):
        x,y = batch
        y = y.view(-1,1)
        y = y.float()
        pred = self.forward(x)
        loss = self.criterion(pred,y)
        k = self.k(pred,y)
        self.log_dict({'training_loss':loss, 'weighted quadratic kappa':k}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y = y.view(-1,1)
        y = y.float()
        pred = self.forward(x)
        loss = self.criterion(pred,y)
        self.log_dict({'val_loss':loss}, prog_bar=True)
        return loss

if __name__ == '__main__':
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=3)
    model = MyNN(lr=0.001)
    data = RetinopathyData(train_images_path,train_csv_path, t = transforms, batch_size=512)
    trainer = Trainer(accelerator='gpu',precision='16-mixed', max_epochs=100, log_every_n_steps=3)
    tuner = Tuner(trainer)
    trainer.fit(model,data)
    #lr_finder = tuner.lr_find(model,data)
    #fig = lr_finder.plot(suggest=True) #0.1
    #plt.show()
    #tuner.scale_batch_size(model,data) #512
    #trainer.fit(model,DataLoader(RetinopathyDataset(train_images_path,train_csv_path,transforms),batch_size=512, num_workers=10))








