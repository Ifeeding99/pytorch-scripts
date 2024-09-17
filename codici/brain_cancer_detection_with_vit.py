import os
from PIL import Image
import torch
import torch.nn as nn
import einops
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import Accuracy
import tqdm

im_path = 'C:/Users/flavi/Downloads/brain_tumors'
img_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.init()

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.mha = nn.MultiheadAttention(self.embed_dim,self.num_heads,self.dropout)
        self.lnorm_1 = nn.LayerNorm(self.embed_dim)
        self.lnorm_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim,self.embed_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self,x):
        input_x = x
        norm_x = self.lnorm_1(x)
        x = input_x + self.mha(norm_x,norm_x,norm_x)[0]
        x = x + self.mlp(self.lnorm_2(x))
        return x

class ViT(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim, patch_size, img_size, n_blocks):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.patch_size = patch_size
        self.img_size = img_size
        assert (self.img_size % self.patch_size == 0), "Image dimension must be divisible by patch dimension!"
        self.n_patches = (self.img_size // self.patch_size)**2

        self.embeddings = nn.Linear(self.patch_size * self.patch_size * 3, self.embed_dim)
        self.class_token = nn.Parameter(torch.Tensor(1,1,self.embed_dim),requires_grad=True)
        self.positional_embeddings = nn.Parameter(torch.Tensor(1,self.n_patches + 1,self.embed_dim), requires_grad=True)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim,self.num_heads,self.dropout,self.hidden_dim+1) for i in range(self.n_blocks)
        ])

        self.fcc = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim,3)

        )


    def forward(self, x):
        # making patches
        x = einops.rearrange(x,'b c (n_h p_h) (n_w p_w) -> b (n_h n_w) (p_h p_w c)', p_h = self.patch_size, p_w = self.patch_size)
        x = self.embeddings(x)
        b_size = x.shape[0]
        c_token = einops.repeat(self.class_token, '1 1 emb -> b 1 emb', b=b_size)
        x = torch.cat((c_token,x), dim = 1)
        pos_encodings = einops.repeat(self.positional_embeddings, '1 n_patch emb -> b n_patch emb', b = b_size)
        x = x + pos_encodings
        for block in self.blocks:
            x = block(x)
        x = self.fcc(x) # shape B N_patches+1 embedding_dim
        output = x[:,0,:] #retrieving the class token
        return output


def get_mean_and_std(loader):
    mean = 0
    std = 0
    n_images = 0
    for image, _ in loader:
        b_size = image.shape[0]
        n_images += b_size
        image = einops.rearrange(image, 'b c h w -> b c (h w)') # flattens image
        im_mean = torch.mean(image.float(), dim=2) # calculates mean in the last dimensio, the result will have shape (B_size, C)
        im_mean = im_mean.sum(0) # sums across axis 0 (batches), the result will be a tensor of shape [C]
        im_std = torch.std(image.float(), dim=2)
        im_std = im_std.sum(0)
        mean += im_mean
        std += im_std

    mean = mean / n_images
    std = std / n_images
    return mean, std

class BrainTumorDataset(Dataset):
    def __init__(self, im_path, t = None):
        # the structure of my training folder is /training/1/images, /training/2/images, /training/3/images
        # the name of the folder containing the images is the class to which the images belong
        self.im_path = im_path
        self.labels = []
        self.images = []
        self.label_classes = os.listdir(self.im_path)
        self.t = t
        for label in self.label_classes:
            for im in os.listdir(os.path.join(self.im_path,label)):
                self.images.append(os.path.join(label,im))
                self.labels.append(int(label)-1) # torch's CrossEntropy loss wants RAW class labels, not onehot encoded labels
        self.l = len(self.labels)


    def __getitem__(self, item):
        image = Image.open(os.path.join(self.im_path,self.images[item]))
        image = image.resize((256,256)) # not enough RAM for wider images
        label = self.labels[item]
        if self.t:
            image = np.array(image) # I am using albumentations, this library expects a numpy array
            # current shape is 512,512,4  the las channels is useless (mean = 255 std = 0)
            image = image[:,:,:3] # slices don't include last index
            augmented = self.t(image=image)
            image = augmented['image']
        return image,label

    def __len__(self):
        return self.l

mean = (62.3742,  49.0386, 112.7380)
std = (10.8123, 46.6892, 23.3724) #these were calculated using get_mean_std and by passing to it the train_loader
t = A.Compose([
    A.Resize(img_size,img_size),
    A.ColorJitter(brightness=(0.2,1), contrast=(0.2,1), saturation=(0.2,1), hue=(-0.5,0.5), p=0.5),
    A.ElasticTransform(p=0.5),
    A.GaussNoise(
    var_limit=(100.0, 500.0),  # Union[float, Tuple[float, float]]
    mean=0,  # float
    per_channel=True,  # bool
    always_apply=False,  # bool
    p=0.7,
),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Sharpen(
    alpha=(0.2, 0.5),  # Tuple[float, float]
    lightness=(0.5, 1.0),  # Tuple[float, float]
    always_apply=False,  # bool
    p=1.0,  # float
),
    A.Normalize(mean=mean, std=std),
    ToTensorV2() # doesn't scale between 0 and 1
])

dataset = BrainTumorDataset(im_path, t)
train_loader = DataLoader(dataset, batch_size=8, num_workers=14, shuffle=True)


MyViT = ViT(300,10,0,200,8,img_size,4)
MyViT = MyViT.to(device)
n_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MyViT.parameters(),lr=1e-3,momentum=0.9,nesterov=True)
acc = Accuracy("multiclass",num_classes=3)
acc = acc.to(device)


if __name__ == '__main__':
    m, s = get_mean_and_std(train_loader)
    for epoch in range(n_epochs):
        loop = tqdm.tqdm(train_loader)
        for batch_idx, (image, label) in enumerate(loop):
            image = image.to(device)
            label = label.to(device)
            pred = MyViT(image)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc.update(pred,label)

            loop.set_description(f'Epoch {epoch+1}/{n_epochs}')
            loop.set_postfix(Loss=loss.item(), accuracy=acc.compute())
        acc.reset()





        



