import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataprocessing import matplotlib_imshow

import dataprocessing


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sequence = nn.Sequential(

        )


    def forward(self, x):
        x = torch.nan_to_num(x, nan=0, neginf=0, posinf=0)


        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

data = dataprocessing.load_data()

data = dataprocessing.make_block(data)

net = NeuralNetwork().to(device)

net.load_state_dict(torch.load("C:/Users/bugsi/PycharmProjects/WaterScarcityProject/model.pt"))
net.eval()

data = dataprocessing.CustomDataset(data, transform=torch.from_numpy, target_transform=torch.from_numpy)

train_dataloader = DataLoader(data, batch_size=64, shuffle=True)

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)

dataiter = iter(train_dataloader)
block, block_pred = dataiter.next()

running_loss = 0.0
for epoch in range(1000):  # loop over the dataset multiple times
    print(epoch)
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        block, block_pred = data

        block = torch.nan_to_num(block.to(device, dtype=torch.float), nan=0, neginf=0, posinf=0)
        block_pred = torch.nan_to_num(block_pred.to(device, dtype=torch.float), nan=0, neginf=0, posinf=0)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(block)
        loss = criterion(outputs, block_pred)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(loss.item())

block, block_pred = dataiter.next()
block = torch.nan_to_num(block.to(device, dtype=torch.float), nan=0, neginf=0, posinf=0)
block_pred = torch.nan_to_num(block_pred.to(device, dtype=torch.float), nan=0, neginf=0, posinf=0)
outputs = net(block)
print(outputs)

fig, axs = plt.subplots(2)
pos1 = axs[0].imshow(block_pred.cpu().detach().numpy()[0].reshape(180, 360), vmin=-0.25, vmax=0.25)
plt.colorbar(pos1, ax=axs[0])
pos2 = axs[1].imshow(outputs.cpu().detach().numpy()[0].reshape(180, 360), vmin=-0.25, vmax=0.25)
plt.colorbar(pos2, ax=axs[1])

torch.save(net.state_dict(), "C:/Users/bugsi/PycharmProjects/WaterScarcityProject/model.pt")

plt.show()

print('Finished Training')


