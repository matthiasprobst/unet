import torch
import torchvision.transforms.functional as TF
from torch import nn


class ConvUp(nn.Module):
    """Convolution + Upsampling"""

    def __init__(self, in_channels, out_channels,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        """Forward method"""
        return self.conv(x)


class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels, N: int,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        if N < 1:
            raise ValueError('N<1 not allowed!')
        convs = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=False),
        ]
        for _ in range(N - 1):
            convs.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False
                ))
            convs.append(
                nn.BatchNorm2d(num_features=out_channels),
            )
            convs.append(
                nn.ReLU(inplace=False)
            )
        self.conv = nn.Sequential(*convs)

    def forward(self, x):
        """Forward method"""
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        """Forward method"""
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self,
            in_channels=1,  # e.g. 1 for gray scale 3 for rgb images
            out_channels=1,
            features=[64, 128, 256, 512],
            up_stride: int = 1,
            down_stride: int = 1,
            up_kernel_size: int = 3,
            down_kernel_size: int = 3,
            pooling_kernel_size: int = 2,
            pooling_stride: int = 2,
            bottleneck_kernel_size: int = 3,
            bottleneck_stride: int = 1,
            use_upsample: bool = False  # False --> original U-NET implementation
    ):
        super().__init__()
        self.use_upsample = use_upsample

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=pooling_kernel_size,
                                 stride=pooling_stride)

        # Down part of UNET
        for feature in features:
            self.downs.append(MultiConv(in_channels,
                                        feature,
                                        2,
                                        down_kernel_size,
                                        down_stride))
            in_channels = feature  # update in_channels

        # bottelneck (lowest part of UNET)
        self.bottleneck = MultiConv(features[-1],
                                    features[-1] * 2,
                                    2,
                                    bottleneck_kernel_size,
                                    bottleneck_stride)

        # Up part of UNET
        for ifeature, feature in enumerate(reversed(features)):
            if ifeature == 0:
                if use_upsample:
                    self.ups.append(
                        ConvUp(
                            in_channels=feature * 2,
                            out_channels=feature,
                            kernel_size=2,
                            stride=2)
                    )
                else:
                    self.ups.append(
                        nn.ConvTranspose2d(
                            in_channels=feature * 2,
                            out_channels=feature,
                            kernel_size=2,
                            stride=2
                        )
                    )
            else:
                if feature == features[::-1][ifeature - 1]:
                    if use_upsample:
                        self.ups.append(
                            ConvUp(
                                in_channels=feature,
                                out_channels=feature,
                                kernel_size=2,
                                stride=2)
                        )
                    else:
                        self.ups.append(
                            nn.ConvTranspose2d(
                                in_channels=feature,
                                out_channels=feature,
                                kernel_size=2,
                                stride=2
                            )
                        )
                else:
                    if use_upsample:
                        self.ups.append(
                            ConvUp(
                                in_channels=feature * 2,
                                out_channels=feature,
                                kernel_size=2,
                                stride=2)
                        )
                    else:
                        self.ups.append(
                            nn.ConvTranspose2d(
                                in_channels=feature * 2,
                                out_channels=feature,
                                kernel_size=2,
                                stride=2
                            )
                        )
            self.ups.append(MultiConv(in_channels=feature * 2,
                                      out_channels=feature,
                                      N=2,
                                      kernel_size=up_kernel_size,
                                      stride=up_stride))

        # the density prediction:
        self.final_conv = nn.Conv2d(features[0],
                                    out_channels,
                                    kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # going down towards the bottelneck
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # original imlementation
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                raise ValueError(f'shapes unequal. Check your net ({x.shape} != {skip_connection.shape}): {self}')
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        # # my implementation:
        # for up1, up2, skip_connection in zip(self.ups[0:-1:2], self.ups[1::2], skip_connections):
        #     x = up1(x)
        #
        #     if x.shape != skip_connection.shape:
        #         x = TF.resize(x, size=skip_connection.shape[2:])
        #
        #     concat_skip = torch.cat((skip_connection, x), dim=1)
        #     x = up2(concat_skip)
        return self.final_conv(x)


def test():
    x = torch.randn(size=(3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1, )
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == '__main__':
    test()
