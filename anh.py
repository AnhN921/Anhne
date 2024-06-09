import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from collections import defaultdict
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 20
learning_rate = 0.0001

# Định nghĩa hàm get_mnist để tải và chuẩn bị dữ liệu MNIST
def get_mnist(train_batch_size, test_batch_size):
    trans_mnist_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trans_mnist_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=trans_mnist_train)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=trans_mnist_test)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader

# Định nghĩa mô hình Lenet
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x = self.fc3(x2)
        protos = x2  # Trích xuất đặc trưng
        log_probs = F.log_softmax(x, dim=1)
        return log_probs, x1

# Định nghĩa hàm train_mnist để huấn luyện mô hình
def train_mnist(args, epochs, train_loader, test_loader, learning_rate):
    model = Lenet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    prototypes = defaultdict(lambda: (torch.zeros(384).to(device), 0))

    for epoch in range(epochs):
        timestart = time.time()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probs, protos = model(inputs)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %4f ' % (epoch, i, running_loss / 500))
                running_loss = 0.0
                _, predicted = torch.max(log_probs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print('Accuracy of the network on the %d train images: %.3f %%' % (total, 100.0 * correct / total))
                total = 0
                correct = 0
            with torch.no_grad():
                for j in range(labels.size(0)):
                    label = labels[j].item()
                    prototype, count = prototypes[label]
                    new_prototype = (prototype * count + protos[j]) / (count + 1)
                    prototypes[label] = (new_prototype, count + 1)
        print('Epoch %d cost %3f sec' % (epoch, time.time() - timestart))
    print('Finish training')
    path = '../save/weights_' + str(epochs) + 'ep.tar'
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path)
    for label, (protos, count) in prototypes.items():
        print(f'Label: {label}, Prototype: {protos}, Count: {count}')
    return prototypes

# Định nghĩa hàm test_mnist để kiểm tra mô hình
def test_mnist(args, test_loader, epochs):
    model = Lenet().to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            log_probs, _ = model(images)
            _, predicted = torch.max(log_probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %3f %%' % (100.0 * correct / total))

# Định nghĩa hàm start_training_task để bắt đầu huấn luyện mô hình
def start_training_task(train_batch_size, test_batch_size):
    train_loader, test_loader = get_mnist(train_batch_size, test_batch_size)
    args = {}  # Thêm các tham số cần thiết vào đây
    prototypes = train_mnist(args, epochs, train_loader, test_loader, learning_rate)
    test_mnist(args, test_loader, epochs)
    return prototypes

# Bắt đầu huấn luyện với batch_size 64 cho cả train và test
start_training_task(64, 64)
