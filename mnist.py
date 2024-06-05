import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import psutil
import json
import random
import pandas as pd
from arg_nene import argsparser #???

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 20
learning_rate = 0.0001
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
#def sampling_noniid_data(df, fraction, num_labels, n):
    label_data_list = []
    # Tổng số mẫu trong DataFrame
    num_samples_total = len(df)
    # Số mẫu cần lấy cho mỗi nhãn
    num_samples_per_label = int(fraction * num_samples_total / num_labels)
    # Chọn ngẫu nhiên 'n' nhãn từ danh sách các nhãn duy nhất trong cột 'type'
    random_labels = random.sample(df['type'].unique().tolist(), n)
    # In các nhãn ngẫu nhiên được chọn
    print("random label: ", random_labels)
    # Lặp qua các nhãn ngẫu nhiên và lấy mẫu dữ liệu cho mỗi nhãn
    for label_name in random_labels:
        print("Cac label: ", label_name)
        # Lọc các hàng trong DataFrame có nhãn là 'label_name'
        label_df = df[df['type'] == label_name]
        # Lấy mẫu ngẫu nhiên 'num_samples_per_label' từ 'label_df'
        label_data = label_df.sample(n=num_samples_per_label, replace=True, random_state=0)
        # Thêm dữ liệu lấy mẫu vào danh sách
        label_data_list.append(label_data)
    # Kết hợp tất cả dữ liệu lấy mẫu thành một DataFrame duy nhất
    final_df = pd.concat(label_data_list)
    
    return final_df

#class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1
    
class Lenet(nn.Module):
    def __init__(self, args):
        super(Lenet, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)
        protos = F.relu(self.fc2(x)) #trichs xuar dac trug
        log_probs = F.log_softmax(self.fc3(protos), dim=1), x1 

        return log_probs, protos 
    
def train_mnist(args, epochs, train_loader, test_loader, learning_rate):
    model = Lenet(args=args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    prototypes = ? 

    for epoch in range(epochs):
        timestart = time.time()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            model.zero_grad() #zero the para grad
            log_probs, protos = model(inputs) #tính đầu ra của model và proto
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %4f ' %
                      (epoch, i, running_loss / 500))
                running_loss = 0.0
                _, predicted = torch.max(log_probs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(' Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0*correct/total))
                total = 0
                correct = 0
            with torch.no_grad():
                for j in range(labels.size(0)):
                    labels = labels[j].item()
                    protos = protos[j]
                    prototype, count = prototypes[labels]
                    new_prototype = (prototype * count + protos)/(count +1) #count = soos maaux dwx lieeju
                    prototypes[labels] = (new_prototype, count + 1)
            print('Epoch %d cost %3f sec' % (epoch, time.time() - timestart))
    print('Finish training')
    path = '../save/weights_'+str(epochs)+'ep.tar'
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path)
    return prototypes
    
def test_mnist(args, test_loader, epochs):
    model = Lenet(args=args)
    model.to(device)
    print('Start testing')
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(args,device), labels.to(args.device)
            log_probs, protos = model(images)
            _, predicted = torch.max(log_probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %3f %%' % (100.0 * correct/total))

def start_training_task(start_line, start_benign):
    lr = learning_rate
    my_df = get_mnist(start_line, start_benign)
    train_loader, test_loader = get_mnist(my_df)
    epochs = epochs 
    model = Lenet().to(device)
    model.load_state_dict(torch.load("newmode.pt", map_location=device))
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.RMSprop(params=model.parameters(), lr=lr)

    # logging.info("Using device: %s", device)

    time_start = time.time()
    # logging.info("\n Time start: %d\n", time_start)

    for epoch in range(epochs):
        # logging.info("\nEpoch: %d\n", epoch+1)
        print(f"\nEpoch: {epoch + 1}")
        train_loss = train_mnist(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                           batch_size=batch_size)
        eval_loss, accuracy = test_mnist(model=model, test_loader=test_loader, criterion=criterion, batch_size=batch_size)
        print(
            "Epoch: {}/{}".format(epoch + 1, epochs),
            "Training Loss: {:.4f}".format(train_loss.item()),
            "Eval Loss: {:.4f}".format(eval_loss),
            "Accuracy: {:.4f}".format(accuracy))
        #ram_usage = psutil.virtual_memory().percent
        #cpu_usage = psutil.cpu_percent()
    print('Finished Training')
    time_end = time.time()
    # logging.info("\n Time end: %d\n", time_end)
    return model.state_dict()
#def save_state_dict(model, path):
    # Lấy các cặp key-value trong state_dict của mô hình và chuyển đổi các giá trị thành list sau khi chuyển về CPU và thành numpy array
    state_dict_as_list = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
    # Mở file tại đường dẫn 'path' trong chế độ ghi ('w')
    with open(path, 'w') as fp:
        # Ghi state_dict dưới dạng JSON vào file
        json.dump(fp=fp, obj=state_dict_as_list)

