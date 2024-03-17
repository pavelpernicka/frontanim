import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder Block: C64-C128-C256-C512-C512-C512-C512-C512
        self.e1 = self.define_encoder_block(3, 64, batchnorm=False)
        self.e2 = self.define_encoder_block(64, 128)
        self.e3 = self.define_encoder_block(128, 256)
        self.e4 = self.define_encoder_block(256, 512)
        self.e5 = self.define_encoder_block(512, 512)
        self.e6 = self.define_encoder_block(512, 512)
        self.e7 = self.define_encoder_block(512, 512)

        # bottlenech, no batch norm, and ReLU
        self.b = nn.Conv2d(512, 512, (4, 4), stride=(2, 2), padding=1)
        nn.init.normal_(self.b.weight, mean=0.0, std=0.02)
        self.actb = nn.ReLU(inplace=True)

        # Decoder block: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        self.d1 = self.define_decoder_block(512, 512)
        self.act1 = nn.ReLU(inplace=True)
        self.d2 = self.define_decoder_block(1024, 512)
        self.act2 = nn.ReLU(inplace=True)
        self.d3 = self.define_decoder_block(1024, 512)
        self.act3 = nn.ReLU(inplace=True)
        self.d4 = self.define_decoder_block(1024, 512, dropout=False)
        self.act4 = nn.ReLU(inplace=True)
        self.d5 = self.define_decoder_block(1024, 256, dropout=False)
        self.act5 = nn.ReLU(inplace=True)
        self.d6 = self.define_decoder_block(512, 128, dropout=False)
        self.act6 = nn.ReLU(inplace=True)
        self.d7 = self.define_decoder_block(256, 64, dropout=False)
        self.act7 = nn.ReLU(inplace=True)

        self.d8 = nn.ConvTranspose2d(
            128, 3, (4, 4), stride=(2, 2), padding=1, bias=False
        )
        nn.init.normal_(self.d8.weight, mean=0.0, std=0.02)
        self.act8 = nn.Tanh()

    def forward(self, x):
        xe1 = self.e1(x)
        xe2 = self.e2(xe1)
        xe3 = self.e3(xe2)
        xe4 = self.e4(xe3)
        xe5 = self.e5(xe4)
        xe6 = self.e6(xe5)
        xe7 = self.e7(xe6)
        b1 = self.actb(self.b(xe7))

        xd8 = self.act1(torch.cat((self.d1(b1), xe7), dim=1))
        xd9 = self.act2(torch.cat((self.d2(xd8), xe6), dim=1))
        xd10 = self.act3(torch.cat((self.d3(xd9), xe5), dim=1))
        xd11 = self.act4(torch.cat((self.d4(xd10), xe4), dim=1))
        xd12 = self.act5(torch.cat((self.d5(xd11), xe3), dim=1))
        xd13 = self.act6(torch.cat((self.d6(xd12), xe2), dim=1))
        xd14 = self.act7(torch.cat((self.d7(xd13), xe1), dim=1))

        xd15 = self.d8(xd14)
        out_image = self.act8(xd15)
        return xd15

    def define_encoder_block(self, in_chan, n_filters, batchnorm=True):
        """Defines an encoder block for the Pix2Pix GAN.
        Args:
             in_chan: The number of input channels.
             n_filters: The number of output channels.
             batchnorm: Whether to use batch normalization.

        Returns:
             The encoder block.
        """
        # Create a list to store the layers of the encoder block
        layers = []

        # Add the convolutional layer with the specified number of in channels and out channels.
        conv_layer = nn.Conv2d(
            in_chan, n_filters, kernel_size=4, stride=2, padding=1, bias=False
        )
        # also initialize the weight of the convulation layer.
        nn.init.normal_(conv_layer.weight, mean=0.0, std=0.02)
        layers.append(conv_layer)

        # Conditionally add batch normalization because it does not require every encoder block
        if batchnorm:
            layers.append(nn.BatchNorm2d(n_filters))

        # Add the LeakyReLU activation
        layers.append(nn.LeakyReLU(0.2))

        # Create a sequential block using the list of layers
        encoder_block = nn.Sequential(*layers)

        return encoder_block

    def define_decoder_block(self, in_chan, out_chan, dropout=True):
        """Defines a decoder block for the Pix2Pix GAN.

        Args:
             in_chan: The number of input channels.
             n_filters: The number of output channels.
             dropout: Whether to use dropout.

        Returns:
             The decoder block.
        """
        # Create a list to store the layers of the decoder block.
        layers = []
        # Add the transposed convolutional layer with the specified number of in channels and out channels.
        g = nn.ConvTranspose2d(
            in_chan, out_chan, (4, 4), stride=(2, 2), padding=1, bias=False
        )
        # Initalize the weight of the ConvtTranspose2d layer.
        nn.init.normal_(g.weight, mean=0.0, std=0.02)
        layers.append(g)
        # Using batch norm for every block
        g = nn.BatchNorm2d(out_chan)
        layers.append(g)
        # Conditionally add a dropout layer because it does not require every decoder block.
        if dropout:
            g = nn.Dropout(0.5)
            layers.append(g)
        return nn.Sequential(*layers)
