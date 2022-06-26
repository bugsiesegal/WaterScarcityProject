import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import os
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F


def load_data() -> list[np.ndarray, float]:
    data = []

    for path in glob(os.path.join(os.getcwd(), "data/*.nc")):
        ds = nc.Dataset(path)
        data.append([np.flip(ds["lwe_thickness"][0, :, :], axis=0), ds["time"][0]])

    data = sorted(data, key=lambda d: d[1])

    return data


def make_block(data: list[np.ndarray, float], block_size=3, pred_dist=1):
    block_data = []

    for i in range(len(data) - block_size - pred_dist):
        block_data.append(
            [np.array([data[i + b][0] for b in range(block_size)]), np.array(data[i + block_size + pred_dist][0])])

    return block_data


class CustomDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.block_images = np.array([d[0] for d in data])
        self.pred = np.array([d[1] for d in data])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.block_images)

    def __getitem__(self, idx):
        block = self.block_images[idx]
        block_pred = self.pred[idx]
        if self.transform:
            block = self.transform(block)
        if self.target_transform:
            block_pred = self.target_transform(block_pred)

        return block, block_pred


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg)
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
