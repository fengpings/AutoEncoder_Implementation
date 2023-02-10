import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils


class AEEncoder(nn.Module):
    def __init__(self, img_size, in_channels, output_features, hidden_dims=None):
        super(AEEncoder, self).__init__()
        if isinstance(img_size, int):
            h, w = img_size, img_size
        else:
            h, w = img_size
        self.output_features = output_features
        if hidden_dims is None:
            # 通道的数目
            hidden_dims = [32, 64, 128]
        if len(hidden_dims) != 3:
            raise ValueError("仅支持三隐层!")
        layers = []
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels, out_channels=h_dim,
                        kernel_size=(3, 3), stride=(2, 2), padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.PReLU()
                )
            )
            in_channels = h_dim
            # 更新feature map的大小
            h = (h - 3 + 2 + 2) // 2
            w = (w - 3 + 2 + 2) // 2
        layers.append(nn.Conv2d(hidden_dims[-1], output_features, (h, w), (h, w), padding=0))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: 原始输入图像: [N,1,28,28]
        :return:
        """
        z = self.encoder(x)
        return z.view(-1, self.output_features)


# noinspection DuplicatedCode
class AEDecoder(nn.Module):
    def __init__(self, input_features, img_channels, hidden_dims=None):
        super(AEDecoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        if len(hidden_dims) != 3:
            raise ValueError("仅支持三隐层!")

        self.decoder_input = nn.Linear(in_features=input_features, out_features=hidden_dims[0] * 16)

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1),
                        output_padding=(1, 1)
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.PReLU()
                )
            )
        layers.append(
            nn.ConvTranspose2d(
                in_channels=hidden_dims[-1],
                out_channels=img_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1)
            )
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(x.shape[0], -1, 4, 4)
        return self.decoder(x)


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = AEEncoder(32, 1, 64)
        self.decoder = AEDecoder(64, 1)

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z

    def generate(self, z):
        z = self.decoder(z)
        return z.view(-1, 1, 32, 32)


if __name__ == '__main__':
    # https://github.com/AntixK/PyTorch-VAE
    _batch_size = 16
    _total_epoch = 100

    # 1. 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
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
        z = torch.from_numpy(np.random.normal(0.0, 1.0, size=(1, 64))).float()
        img3 = net.generate(z)  # 基于随机数生成图像
        img = torch.cat([img, img2, img3], dim=0)
        _dir = f'./output/ae2/image'
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        utils.save_image(img, f'{_dir}/{epoch}_{label}.png')

    torch.save(net, './output/ae2/model.pth')
