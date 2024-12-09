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
        self.encoder_layers = nn.ModuleList([
            build_conv_block(in_channels, 64, 3, 1, 1, 'relu'),
            build_conv_block(64, 128, 3, 2, 1, 'relu'),
            build_conv_block(128, 256, 3, 2, 1, 'relu'),
            build_conv_block(256, 512, 3, 2, 1, 'relu'),
            build_conv_block(512, 512, 3, 2, 1, 'relu')
        ])

    def forward(self, x):
        features = []
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        return features

# 解码器
class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([
            build_deconv_block(512, 256, 2, 2, 0, 'relu'),
            build_deconv_block(256 + 512, 128, 2, 2, 0, 'relu'),
            build_deconv_block(128 + 256, 64, 2, 2, 0, 'relu'),
            build_deconv_block(64 + 128, 64, 2, 2, 0, 'relu'),
            build_deconv_block(64 + 64, out_channels, 3, 1, 1, 'sigmoid')
        ])

    def forward(self, features):
        x = features[-1]
        for i, layer in enumerate(self.decoder_layers):
            if i > 0:
                x = torch.cat([x, features[-(i+2)]], dim=1)
            x = layer(x)
        return x

# 生成器
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, data, label):
        features = self.encoder(torch.cat([data, label], dim=1))
        return self.decoder(features)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.disc_layers = nn.ModuleList([
            build_conv_block(in_channels, 64, 3, 1, 1, 'relu'),
            build_conv_block(64, 128, 3, 2, 1, 'relu'),
            build_conv_block(128, 256, 3, 2, 1, 'relu'),
            build_conv_block(256, 512, 3, 2, 1, 'relu'),
            build_conv_block(512, 512, 3, 2, 1, 'relu')
        ])
        self.out = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1200, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, data, label):
        x = torch.cat([data, label], dim=1)
        for layer in self.disc_layers:
            x = layer(x)
        x = x.view(-1, 512 * 4 * 4)
        return self.out(x)
