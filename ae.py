import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils


class AEEncoder(nn.Module):
    def __init__(self, input_features, output_features, hidden_dims=None):
        super(AEEncoder, self).__init__()
        self.input_features = input_features
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        layers = []
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features=input_features, out_features=h_dim),
                    nn.BatchNorm1d(num_features=h_dim),
                    nn.PReLU()
                )
            )
            input_features = h_dim
        layers.append(nn.Linear(input_features, output_features))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: 原始输入图像: [N,1,28,28]
        :return:
        """
        x = x.view(-1, self.input_features)
        return self.encoder(x)


class AEDecoder(nn.Module):
    def __init__(self, input_features, output_features, hidden_dims=None):
        super(AEDecoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        layers = []
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features=input_features, out_features=h_dim),
                    nn.BatchNorm1d(num_features=h_dim),
                    nn.PReLU()
                )
            )
            input_features = h_dim
        layers.append(nn.Linear(input_features, output_features))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = AEEncoder(784, 32)
        self.decoder = AEDecoder(32, 784)

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z.view(-1, 1, 28, 28)

    def generate(self, z):
        z = self.decoder(z)
        return z.view(-1, 1, 28, 28)


if __name__ == '__main__':
    _batch_size = 16
    _total_epoch = 100

    # 1. 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=(28, 28), scale=(0.85, 1.0))
    ])
    trainset = datasets.MNIST(root='../data/MNIST', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=_batch_size, shuffle=True, num_workers=2, prefetch_factor=_batch_size)
    testset = datasets.MNIST(root='../data/MNIST', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=_batch_size, shuffle=True, num_workers=2, prefetch_factor=_batch_size)

    # 2. 模型加载
    net = AE()
    loss_fn = nn.MSELoss()
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.7)

    # writer = SummaryWriter(log_dir='./output/ae')
    # writer.add_graph(net, torch.empty(10, 1, 28, 28))

    # 3. 开始模型训练
    total_samples = len(trainset)
    total_epoch = _total_epoch
    summary_step_interval = 200
    train_step = 0
    test_step = 0
    for epoch in range(total_epoch):
        # 训练操作
        net.train(True)
        train_loss = []
        for data in trainloader:
            inputs, _ = data

            # 前向过程
            outputs = net(inputs)
            _loss = loss_fn(outputs, inputs)
            # 反向过程
            opt.zero_grad()
            _loss.backward()
            opt.step()

            train_loss.append(_loss.item())
            if train_step % summary_step_interval == 0:
                # 可视化输出
                print(f"Train {epoch + 1}/{total_epoch} {train_step}  loss:{_loss.item():.3f}")
            train_step += 1

        # 测试操作
        net.eval()
        test_loss = []
        for data in testloader:
            inputs, _ = data

            # 前向过程
            outputs = net(inputs)
            _loss = loss_fn(outputs, inputs)

            test_loss.append(_loss.item())
            if test_step % summary_step_interval == 0:
                # 可视化输出
                print(f"Test {epoch + 1}/{total_epoch} {test_step}  loss:{_loss.item():.3f}")
            test_step += 1

        # 做一个随机的预测
        net.eval()
        idx = np.random.randint(0, len(testset))
        img, label = testset[idx]
        img = img[None, ...]
        img2 = net(img)
        z = torch.from_numpy(np.random.normal(0.0, 1.0, size=(1, 32))).float()
        img3 = net.generate(z)  # 基于随机数生成图像
        img = torch.cat([img, img2, img3], dim=0)
        _dir = f'./output/ae/image'
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        utils.save_image(img, f'{_dir}/{epoch}_{label}.png')

    torch.save(net, './output/ae/model.pth')
