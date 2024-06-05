import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 20

def get_mnist(train_batch_size, test_batch_size):
    # Define data transformations
    trans_mnist_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trans_mnist_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=trans_mnist_train)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=trans_mnist_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader
