import torch
import torch.nn as nn
import torch.optim as optim
import  torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


batch_size = 100
original_dim = 784
latent_dim = 2 # 隐变量取2维只是为了方便后面画图
intermediate_dim = 256
epochs = 50

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):

    def __init__(self,original_dim,latent_dim,intermediate_dim):
         super(VAE,self).__init__()
         self.original_dim=original_dim
         self.latent_dim=latent_dim
         self.intermediate_dim=intermediate_dim
         self.hidden_dim=nn.Linear(self.original_dim,self.intermediate_dim)

         self.mean=nn.Linear(self.intermediate_dim,self.latent_dim)
         self.log_var=nn.Linear(self.intermediate_dim,self.latent_dim)


         self.decoder_h=nn.Linear(self.latent_dim,self.intermediate_dim)
         self.decoder_mean=nn.Linear(self.intermediate_dim,self.original_dim)


    def sample(self,args):
         #均值和反差的log
         z_mean,z_log_var=args
         #从标准正态分布采样
         epsilon=torch.rand_like(z_mean)
         return z_mean+torch.exp(z_log_var/2)*epsilon

    def forward(self,x):
      output=F.relu(self.hidden_dim(x))
      z_mean=self.mean(output)
      z_log_var=self.log_var(output)
      output=self.sample((z_mean,z_log_var))
      output=F.relu(self.decoder_h(output))
      output=F.sigmoid(self.decoder_mean(output))
      return output

model=VAE(original_dim,latent_dim,intermediate_dim).to(device)

criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(epochs):
 for data in train_loader:
               img,_ = data
               img=img.to(device)
               img=img.view(img.size(0),-1)

               output=model(img)
               loss=criterion(output,img)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
 print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
















