import torch
import torch.nn as nn
from einops import rearrange

# 定义卷积块，包括卷积、BN和激活函数
def build_conv_block(in_channels, out_channels, kernel_size, stride=1, padding=1, activation='relu'):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels)
    ]
    if activation == 'relu':
        layers.append(nn.LeakyReLU())
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

# 定义转置卷积块，包括转置卷积、BN和激活函数
def build_deconv_block(in_channels, out_channels, kernel_size, stride=1, padding=1, activation='relu'):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels)
    ]
    if activation == 'relu':
        layers.append(nn.LeakyReLU())
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

# 编码器
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.channels = in_channels
        self.conv_stacks = nn.ModuleList([
            build_conv_block(self.channels*2, 16, 6, 2, 2, 'relu'),
            build_conv_block(16, 16, 3, 'relu'),
            build_conv_block(16, 64, 6, 2, 2, 'relu'),
            build_conv_block(64, 64, 3, 'relu'),
            build_conv_block(64, 256, 6, 2, 2, 'relu'),
            build_conv_block(256, 256, 3, 'relu'),
            build_conv_block(256, 512, 6, 2, 2, 'relu'),
            build_conv_block(512, 512, 3, 'relu')
        ])
        self.downres = build_conv_block(self.channels*2, 512, 24, 16, 4, 'relu')

    def forward(self, data, label):
        x = torch.cat([data, label], dim=1)
        outputs = [x]
        for conv_stack in self.conv_stacks:
            x = conv_stack(x)
            outputs.append(x)
        x = outputs[-1] + self.downres(x)
        return outputs + [x]

# 解码器
class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.deconv_blocks = nn.ModuleList([
            build_deconv_block(512, 800, 3, 1, 1, 'relu'),
            build_deconv_block(800, 800, 6, 2, 2, 'relu'),
            build_deconv_block(256+800, 320, 6, 2, 2, 'relu'),
            build_deconv_block(320+64, 90, 6, 2, 2, 'relu'),
            build_deconv_block(90+16, 44, 6, 2, 2, 'relu')
        ])
        self.out = nn.Sequential(
            build_deconv_block(44+6, 24, 3, 1, 1, 'relu'),
            build_deconv_block(24, 8, 3, 1, 1, 'relu'),
            build_deconv_block(8, out_channels, 1, 1, 0, 'sigmoid')
        )

    def forward(self, *features):
        x = features[0]
        for i in range(len(self.deconv_blocks)):
            if i > 0:
                x = torch.cat([x, features[i+1]], dim=1)
            x = self.deconv_blocks[i](x)
        return self.out(torch.cat([x, features[-1]], dim=1))

# 生成器
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, data, label):
        features = self.encoder(data, label)
        return self.decoder(*features)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(in_channels)
        self.down_blocks = nn.ModuleList([
            nn.Sequential(build_conv_block(16, 4, 6, 2, 2, 'relu'), nn.MaxPool2d(4, 4), build_conv_block(4, 4, 3, 'relu')),
            nn.Sequential(build_conv_block(64, 8, 6, 2, 2, 'relu'), nn.MaxPool2d(2, 2), build_conv_block(8, 8, 3, 'relu')),
            nn.Sequential(build_deconv_block(256, 16, 6, 2, 2, 'relu'), nn.MaxPool2d(2, 2), build_conv_block(16, 16, 3, 'relu'), nn.MaxPool2d(2, 2)),
            nn.Sequential(build_deconv_block(512, 200, 6, 2, 2, 'relu'), nn.MaxPool2d(2, 2), build_deconv_block(200, 100, 6, 2, 2, 'relu'), nn.MaxPool2d(2, 2), build_deconv_block(100, 50, 6, 2, 2, 'relu'), nn.MaxPool2d(2, 2))
        ])
        self.out = nn.Sequential(
            nn.Linear(78*16*16, 1200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1200, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, data, label):
        features = self.encoder(data, label)
        downs = [down_block(feature) for down_block, feature in zip(self.down_blocks, features[1:])]
        down = torch.cat(downs, dim=1)
        down = down.view(-1, 78*16*16)
        return self.out(down)
