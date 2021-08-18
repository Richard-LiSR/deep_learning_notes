import torch
from model import LeNet
from torch.utils.data import DataLoader
import torchvision

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

cnn = LeNet()  # 一定要记得实例化对象！！
cnn.load_state_dict(torch.load('../weights/mnist.pth'))
print(cnn)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = cnn.to(device)  # 一定要将加载好的模型放到gpu中
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor) / 255.
test_x = test_x.cuda()  # 将数据放在GPU中
test_y = test_data.targets
test_output = cnn(test_x[:10])[0]
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()  # 注意：GPU中的数据不能直接转换成numpy格式，需要将其转换成cpu再进行转换
print(pred_y, 'prediction number')
print(test_data.targets[:10].numpy(), 'real number')
