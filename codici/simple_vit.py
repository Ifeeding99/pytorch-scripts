import os
import einops
from PIL import Image
import torch
import torch.nn as nn
import tqdm
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, DataLoader, random_split

train_folder_path ='C:/Users/flavi/Downloads/dogs-vs-cats/train'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class C_D_Dataset(Dataset):
    def __init__(self, path, transforms=None):
        super().__init__()
        self.path = path
        self.transforms = transforms
        self.list_images = os.listdir(self.path)
        self.l = len(self.list_images)
        self.labels = []
        for i in range(self.l):
            if 'cat' in self.list_images[i]:
                self.labels.append(0)
            else:
                self.labels.append(1)

    def __getitem__(self, item):
        im = Image.open(os.path.join(self.path, self.list_images[item]))
        label = self.labels[item]
        if self.transforms:
            im = self.transforms(im)
        return im,label

    def __len__(self):
        return self.l

def patchify(img, patch_size, flatten_channels = True):
    img = img.unfold(2, patch_size, patch_size)# N C H W -> N C H//p_size, W
    img = img.unfold(3, patch_size, patch_size)# N C p_size p_size p_size p_size
    img = img.contiguous() # to dodge potential errors
    img = img.reshape(-1,3,patch_size,patch_size)
    if flatten_channels:
        img = img.flatten(1)
    else:
        img = img.flatten(2)
    return img



class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.l_norm_1 = nn.LayerNorm(self.embed_dim)
        self.mha = nn.MultiheadAttention(self.embed_dim, self.n_heads)
        self.l_norm_2 = nn.LayerNorm(self.embed_dim)
        self.fcc = nn.Sequential(
            nn.Linear(self.embed_dim, 300),
            nn.GELU(),
            nn.Linear(300, self.embed_dim)
        )

    def forward(self,x):
        norm_x = self.l_norm_1(x)
        x = x + self.mha(norm_x, norm_x, norm_x)[0]
        x = x + self.fcc(self.l_norm_2(x))
        return x

class MyViT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, img_size, patch_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (self.img_size // self.patch_size)**2
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.embeddings = nn.Linear(self.patch_size * self.patch_size * 3, self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1,1,self.embed_dim))
        self.positionals_embeddings = nn.Parameter(torch.randn(1,self.num_patches+1,self.embed_dim)) # +1 for the cls token
        self.Transformers_Blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads) for i in range(self.num_blocks)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,1),
            nn.Sigmoid()
        )


    def forward(self, x):
        # x = patchify(x,self.patch_size)
        # or, using einops
        x = einops.rearrange(x,'b c (n_h p_h) (n_w p_w) -> b (n_h n_w) (p_h p_w c)',p_h=self.patch_size, p_w=self.patch_size)
        x = self.embeddings(x)
        batch_size = x.shape[0]
        cls = einops.repeat(self.cls_token,'1 1 emb -> batch 1 emb', batch = batch_size) # to repeat and introduce a new dimension or expand
        x = torch.cat((cls, x), dim=1)
        x = x + self.positionals_embeddings[:,:x.shape[1]] # the sum has to occur in the dimension 1
        for i,layer in enumerate(self.Transformers_Blocks):
            x = self.Transformers_Blocks[i](x)
        x = self.mlp(x)
        return x[:,0,:] # this is the cls token

t = Compose([
    Resize((256,256)),
    ToTensor()
])

dataset = C_D_Dataset(train_folder_path, transforms=t)
train_examples = int(0.8*len(dataset))
train_dataset, val_dataset = random_split(dataset,[train_examples, len(dataset)-train_examples])
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=32,num_workers=10)
val_loader = DataLoader(val_dataset,shuffle=True,batch_size=32,num_workers=10)


ViT = MyViT(700, 10,5,256,8)
ViT = ViT.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(ViT.parameters(),lr=1e-3,nesterov=True,momentum=0.9)
n_epochs = 10

if __name__ == '__main__':
    print('starting training')
    for i in range(n_epochs):
        print(f'Epoch {i+1}/{n_epochs}')
        loop = tqdm.tqdm(train_loader)
        for batch,(x,y) in enumerate(loop):
            x = x.to(device)
            y = einops.rearrange(y,'n -> n 1')
            y = y.float()
            y = y.to(device)
            pred = ViT(x)
            loss = criterion(pred,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(loss.item())

