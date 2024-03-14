import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Weight initialization
        self.conv1 = nn.Conv2d(6, 64, (4, 4), stride=(2, 2), padding=1, bias=False)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv2 = nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)

        self.conv3 = nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(negative_slope=0.2)

        self.conv4 = nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(negative_slope=0.2)

        self.conv5 = nn.Conv2d(512, 512, (4, 4), padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(512)
        self.act5 = nn.LeakyReLU(negative_slope=0.2)

        self.conv6 = nn.Conv2d(512, 3, (4, 4), padding=1, bias=False)
        self.patch_out = nn.Sigmoid()

        # weight initializer all conv2d layer
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, s_img, t_img):

        # Concatenate source image and target image
        m_img = torch.cat((s_img, t_img), dim=1)
        # C64: 4x4 kernel stride 2x2
        x = self.act1(self.conv1(m_img))
        # C128: 4x4 kernel stride 2x2
        x = self.act2(self.bnorm1(self.conv2(x)))
        # C256: 4x4 kernel stride 2x2
        x = self.act3(self.bnorm2(self.conv3(x)))
        # C512: 4x4 kernel stride 2x2
        x = self.act4(self.bnorm3(self.conv4(x)))
        # C512: 4x4 kernel stride 2x2
        x = self.act5(self.bnorm4(self.conv5(x)))
        # Patch Output
        x = self.patch_out(self.conv6(x))
        return x
