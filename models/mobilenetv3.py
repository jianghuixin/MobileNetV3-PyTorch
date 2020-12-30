"""
MobileNetV3 实现
"""
import torch
from torch import nn


class SELayer(nn.Module):
    """
    Squeeze-and-Excite
    """

    def __init__(self, in_channel, reduction=4):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        B, C, _, _ = x.shape

        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)

        return x * y.expand_as(x)


class BottleNeck(nn.Module):
    def __init__(self, inp, expand, oup, kernel, stride, has_se, nonlinear):
        super().__init__()

        self.has_res_connect = (stride == 1) and (inp == oup)

        padding = (kernel - 1) // 2

        if nonlinear == "RE":
            Active = nn.ReLU
        elif nonlinear == "HS":
            Active = nn.Hardswish
        else:
            raise NotImplementedError

        blocks = []

        blocks.append(
            nn.Sequential(
                nn.Conv2d(inp, expand, 1, 1, 0, bias=False),
                nn.BatchNorm2d(expand),
                Active(inplace=True),
            )
        )

        if has_se:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(expand, expand, kernel, stride, padding, groups=expand, bias=False),
                    nn.BatchNorm2d(expand),
                    SELayer(expand),
                    Active(inplace=True),
                )
            )
        else:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(expand, expand, kernel, stride, padding, groups=expand, bias=False),
                    nn.BatchNorm2d(expand),
                    Active(inplace=True),
                )
            )

        blocks.append(
            nn.Sequential(
                nn.Conv2d(expand, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        )

        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        if self.has_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, classes=1000, mode="large"):
        super().__init__()

        if mode == "large":
            mobile_settings = [
                # k, exp, oup, SE, NL, s
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1],
            ]
        else:
            raise NotImplementedError

        features = []
        # input channel to main blocks
        in_channel = 16

        # building the first layer
        features.append(
            nn.Sequential(
                nn.Conv2d(3, in_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.Hardswish(inplace=True),
            )
        )

        # building mobile bottleneck blocks
        for k, exp_channel, out_channel, SE, NL, s in mobile_settings:
            features.append(BottleNeck(in_channel, exp_channel, out_channel, k, s, SE, NL))
            in_channel = out_channel

        # building the last several layers
        last_conv_channel = 960
        last_channel = 1280
        if mode == "large":
            features.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, last_conv_channel, 1, 1, bias=False),
                    nn.BatchNorm2d(last_conv_channel),
                    nn.Hardswish(inplace=True)
                )
            )

            features.append(nn.AdaptiveAvgPool2d(1))
            features.append(
                nn.Sequential(
                    nn.Conv2d(last_conv_channel, last_channel, 1, 1, bias=False),
                    nn.Hardswish(inplace=True)
                )
            )

        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(last_channel, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[:2])
        r = self.classifier(x)

        return r


if __name__ == "__main__":
    # 测试输出维度
    model = MobileNetV3()

    inp = torch.randn(4, 3, 224, 224)

    oup = model(inp)
    # (4, 1000)
    print(oup.shape)
