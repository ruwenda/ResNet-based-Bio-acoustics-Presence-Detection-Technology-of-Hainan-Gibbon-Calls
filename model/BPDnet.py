from torch import nn
from torchlibrosa import SpecAugmentation

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)

        if downsample is not None:
            self.bn1 = nn.BatchNorm2d(in_channel)
        else:
            self.bn1 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))

        return x + identity


class BPDnet(nn.Module):
    def __init__(self, first_channel=16, isSpecAugmentation=False):
        """
        BPDnet model
        :param first_channel:  the parameter in paper to control the number of model parameters
        :param isSpecAugmentation: to use the SpecAugmentation or not
        """
        super(BPDnet, self).__init__()
        self.isSpecAugmentation = isSpecAugmentation
        if self.isSpecAugmentation:
            self.specAugmentation = SpecAugmentation(6, 2, 9, 2)
        self.conv1 = nn.Conv2d(1, first_channel, 7, 2)
        self.bn = nn.BatchNorm2d(first_channel)
        self.maxPool = nn.MaxPool2d(3, 2, padding=1)
        self.conv2_x = nn.Sequential(BasicBlock(first_channel, first_channel * 2, stride=2,
                                                downsample=nn.Conv2d(first_channel, first_channel * 2, 1, 2)),
                                     BasicBlock(first_channel * 2, first_channel * 2))
        self.conv3_x = nn.Sequential(BasicBlock(first_channel * 2, first_channel * 4, 2,
                                                downsample=nn.Conv2d(first_channel * 2, first_channel * 4, 1, 2)),
                                     BasicBlock(first_channel * 4, first_channel * 4))
        self.conv4 = nn.Conv2d(first_channel * 4, first_channel * 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(first_channel * 4)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(first_channel * 2, first_channel)
        self.linear2 = nn.Linear(first_channel, 2)
        self.thanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if self.training and self.isSpecAugmentation:
            x = self.specAugmentation(x)        # 谱增广
        x = self.maxPool(self.relu(self.bn(self.conv1(x))))
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.relu(self.bn2(x))
        x = self.conv4(x)
        x = self.flatten(self.GAP(x))
        x = self.dropout(x)
        x = self.thanh(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x




















