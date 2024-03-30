import torch
import torch.nn as nn
from generator import ConvBlock

class Discriminator(nn.Module):
    def __init__(self, input_dim=6, num_filter=64, output_dim=1):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, stride=1)
        self.conv5 = nn.Conv2d(num_filter * 8, output_dim, kernel_size=4, stride=1, padding=1)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.sigmoid(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean, std)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)