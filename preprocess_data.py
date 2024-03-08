from scipy.io import loadmat
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load the dataset
data = loadmat('./data/frey_rawface.mat')['ff'].T  # Adjust according to the correct key if needed
data = np.reshape(data, (-1, 28, 20))  # Reshape data assuming the images are 28x20
data = torch.tensor(data, dtype=torch.float32) / 255.  # Normalize to [0, 1]

# Transform and create DataLoader
transform = transforms.Compose([transforms.ToTensor()])
dataset = TensorDataset(data)
