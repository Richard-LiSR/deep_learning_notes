import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from model import LeNet

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 获得数据
train_data = torchvision.datasets.FashionMNIST(
    root='./Fashion-mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_data = torchvision.datasets.FashionMNIST(
    root='./Fashion-mnist',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)

# 加载数据
train_loader = DataLoader(dataset=train_data, num_workers=0, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, num_workers=0, batch_size=32, shuffle=True)


def get_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


# # 对数据进行预览
# print(train_data.data.size()) #查看数据尺寸
# print(train_data.targets[0].numpy()) #查看数据标签
# print(train_data.data.dim())
# # 打印一个批次
# images, labels = next(iter(train_loader))
# img = torchvision.utils.make_grid(images)
# img = img.numpy().transpose(1, 2, 0)
# print(labels)
# plt.imshow(img)
# # plt.show()
# 模型训练和优化
LeNet.train()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LeNet = LeNet.to(device)
optimizer = torch.optim.Adam(LeNet.parameters(), lr=0.001)
criteria = torch.nn.CrossEntropyLoss()  # 在使用损失函数的时候应该先实例化然后再调用
epoch = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for epoch in range(epoch):
    running_loss = 0.0
    running_correct = 0
    print("{}/{}".format(epoch, epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        # print(data)
        target = target.to(device)
        outputs = LeNet(data)[0]
        # print(outputs.shape)
        pred = torch.max(outputs, 1)[1]
        optimizer.zero_grad()
        loss = criteria(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == target)
    testing_correct = 0
    for batch_idx, (data_test, target_test) in enumerate(test_loader):
        data_test = get_cuda(data_test)
        target_test = get_cuda(target_test)
        outputs = LeNet(data_test)[0]
        pred = torch.max(outputs, 1)[1]
        testing_correct += torch.sum(pred == target_test)
    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}%".format(
        running_loss / len(train_data), 100 * running_correct / len(train_data),
        100 * testing_correct / len(test_data)))
torch.save(LeNet.state_dict(), './weights/mnist.pth')  # 使用这种方式保存模型，使用其他模型保存方式会出现模型预测出现问题
