import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Định nghĩa một mạng nơ-ron đơn giản
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(784, 10)  # 28x28 pixels = 784, 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten input image
        x = self.fc(x)
        return x

# Hàm huấn luyện mô hình
def train(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Hàm in ra prototype của mô hình
def print_prototypes(model):
    prototypes = model.fc.weight
    print("Prototypes:")
    print(prototypes)

# Load và tiền xử lý dữ liệu MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Khởi tạo mô hình, bộ tối ưu và hàm mất mát
model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Huấn luyện mô hình
train(model, train_loader, optimizer, criterion)

# In ra prototype
print_prototypes(model)
