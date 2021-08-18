import json
import os.path
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torchvision.models.resnet
from resnet import resnet34


def main():
    # 训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {}".format(device))
    # 对花朵数据集进行预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    #    准备数据集
    # ./ 表示当前目录
    # ../ 表示父级目录
    # ../.. 表示祖父目录

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # 父级工作目录
    image_path = os.path.join(data_root, "data-set", "flower_data")  # 将目录拼接起来
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 制造训练集和验证集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    train_num = len(train_dataset)  # 训练集数量大小

    flower_list = train_dataset.class_to_idx  # 返回其类别索引
    cla_dict = dict(
        (val, key) for key, val in flower_list.items())  # items() 方法把字典中每对 key 和 value 组成一个元组，并把这些元组放在列表中返回。
    # 将其写入json文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', "w") as json_file:
        json_file.write(json_str)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    validate_loader = DataLoader(validate_dataset, batch_size=16, shuffle=True, num_workers=0)
    # dataloader本质是一个可迭代对象，使用iter()访问，不能使用next()访问；
    # 使用iter(dataloader)返回的是一个迭代器，然后可以使用next访问
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 实例化模型
    # 预训练
    net = resnet34()
    # 加载预训练权重
    # load pretrain weights
    model_weight_path = "./resnet34.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # define loss function
    loss_func = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 3
    save_path = '../../weights/resNet34.pth'

    best_acc = 0.0
    train_step = len(train_loader)
    for epoch in range(epochs):
        #     train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_func(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs,
                                                                     loss)
            # train_bar.desc = … 的操作是为了能够看到训练的实施进度
            # % f保留小数点后面六位有效数字， % .3f保留三位小数

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predicted_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predicted_y, val_labels.to(device)).sum().item()
                # 利用输出outputs的预测的标签和真实标签进行比较，相等为1，不同为零,进行累加最后除以验证集样本数

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (
                epoch + 1, running_loss / train_step, val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
    print("Finished Training")


if __name__ == '__main__':
    main()
