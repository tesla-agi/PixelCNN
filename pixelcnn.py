import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_dataset=datasets.MNIST(root="./data",train=True,transform=transforms.ToTensor(),download=True)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle=True)

class MaskedConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,masked_type):
        super(MaskedConv,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,padding=kernel_size//2)
        center=kernel_size//2
        mask=torch.zeros(kernel_size,kernel_size)

        for i in range(kernel_size):
            for j in range(kernel_size):
                if i<center:
                    mask[i][j]=1
                elif i==center and j<center:
                    mask[i][j]=1
                elif i==center and j==center and masked_type=="B":
                    mask[i][j]=1

        self.register_buffer('mask',mask.view(1,1,kernel_size,kernel_size))

    def forward(self,x):
        self.conv.weight.data*=self.mask
        return self.conv(x)


class PixelCNN(nn.Module):
    def __init__(self,):
        super(PixelCNN,self).__init__()

        self.layer1=MaskedConv(1,64,7,"A")
        self.layer2=MaskedConv(64,64,3,"B")
        self.layer3=MaskedConv(64,64,3,"B")
        self.layer4=MaskedConv(64,64,3,"B")
        self.layer5=MaskedConv(64,64,3,"B")
        self.layer6=MaskedConv(64,64,3,"B")
        self.layer7=MaskedConv(64,64,3,"B")
        self.layer8=MaskedConv(64,64,3,"B")
        self.output=nn.Conv2d(64,1,1,)

    def forward(self,x):
        h=F.relu(self.layer1(x))
        h=F.relu(self.layer2(h))
        h=F.relu(self.layer3(h))
        h=F.relu(self.layer4(h))
        h=F.relu(self.layer5(h))
        h=F.relu(self.layer6(h))
        h=F.relu(self.layer7(h))
        h=F.relu(self.layer8(h))
        output=F.sigmoid(self.output(h))

        return output

model=PixelCNN()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(20):
    total_loss=0
    for img,label in train_dataloader:
        x=(img>0.5).float()
        output=model(x)
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









