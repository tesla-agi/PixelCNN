import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_dataset=datasets.MNIST(root='./data_gated_pixel_cnn',train=True,transform=transforms.ToTensor(),download=True)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle=True)

def gated_activation(x):
  first_half,second_half=torch.chunk(x,2,dim=1)
  return torch.tanh(first_half)*torch.sigmoid(second_half)


class MaskedConv1(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size):
    super(MaskedConv1,self).__init__()

    self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=kernel_size//2)

    center=kernel_size//2
    mask=torch.zeros(kernel_size,kernel_size)
    for i in range(kernel_size):
      for j in range(kernel_size):
        if i<center:
          mask[i][j]=1

    self.register_buffer("mask",mask.view(1, 1, kernel_size, kernel_size).expand_as(self.conv.weight).clone())

  def forward(self,x):
    self.conv.weight.data*=self.mask
    return self.conv(x)


class MaskedConv2(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size):
    super(MaskedConv2,self).__init__()

    self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=kernel_size//2)
    center=kernel_size//2
    mask=torch.zeros(kernel_size,kernel_size)
    for i in range(kernel_size):
      for j in range(kernel_size):
        if i==center and j<=center:
          mask[i][j]=1

    self.register_buffer("mask",mask.view(1, 1, kernel_size, kernel_size).expand_as(self.conv.weight).clone())

  def forward(self,x):
    self.conv.weight.data*=self.mask
    return self.conv(x)


class GatedLayer(nn.Module):
  def __init__(self,channels):
    super(GatedLayer,self).__init__()

    self.v_inp=MaskedConv1(channels,2*channels,kernel_size=3)
    self.h_inp=MaskedConv2(channels,2*channels,kernel_size=3)
    self.v_h=nn.Conv2d(2*channels,2*channels,kernel_size=1)
    self.h_res=nn.Conv2d(channels,channels,kernel_size=1)

  def forward(self,v_input,h_input):
    v_features=self.v_inp(v_input)
    v_out=gated_activation(v_features)
    h_features=self.h_inp(h_input)
    h_out=gated_activation(self.v_h(v_features)+h_features)
    h_out=self.h_res(h_out)+h_input

    return v_out,h_out


class PixelCNN(nn.Module):
  def __init__(self, channels=64):
    super(PixelCNN, self).__init__()

    self.v_init = MaskedConv1(1, 2 * channels, 7)
    self.h_init = MaskedConv2(1, channels, 3)
    self.layers = nn.ModuleList([
      GatedLayer(channels)
      for _ in range(7)
    ])
    self.out1 = nn.Conv2d(channels, channels, 1)
    self.out2 = nn.Conv2d(channels, 1, 1)

  def forward(self, x):
    v = gated_activation(self.v_init(x))
    h = self.h_init(x)

    for layer in self.layers:
      v, h = layer(v, h)

    out1 = F.relu(self.out1(h))
    out2 = F.sigmoid(self.out2(out1))

    return out2

model=PixelCNN()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(3):
  total_loss=0
  for img,label in train_dataloader:
    x=(img>0.5).float()
    output = model(x)
    loss=F.binary_cross_entropy(output,x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss+=loss.item()

  print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")


def sample(model,num_samples=10):
  model.eval()
  with torch.no_grad():
    x=torch.zeros(num_samples,1,28,28)
    for i in range(784):
      out=model(x)
      row=i//28
      col=i%28
      prob=out[:,0,row,col]
      x[:,0,row,col]=torch.bernoulli(prob)

  return x


samples = sample(model, num_samples=16)
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(16):
    ax = axes[i // 4][i % 4]
    ax.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
    ax.axis('off')

plt.suptitle('PixelCNN Samples')
plt.tight_layout()
plt.savefig('pixelcnn_samples.png')
plt.show()