import torch.nn as nn


# Backup
class PartSegHead(nn.Module):
    def __init__(self, cfg):
        super(PartSegHead, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=cfg['in_c'],
            out_channels=cfg['mid_c'],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(cfg['mid_c'])
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=cfg['mid_c'],
            out_channels=cfg['num_classes'],
            kernel_size=1,
            stride=1,
            padding=0,
        )
        nn.init.normal_(self.deconv.weight, std=0.001)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(self.deconv(x))))
        return x


class PartSegHeadConv(nn.Module):
    def __init__(self, cfg):
        super(PartSegHeadConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=cfg['in_c'], out_channels=cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        return x


class PartSegHeadConvConv(nn.Module):
    def __init__(self, cfg):
        super(PartSegHeadConvConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=cfg['in_c'], out_channels=cfg['mid_c'], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg['mid_c'])
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=cfg['mid_c'], out_channels=cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.conv1.weight, std=0.001)
        nn.init.normal_(self.conv2.weight, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        x = self.conv2(self.relu1(self.bn1(self.conv1(x))))
        return x


class PartSegHeadDeconvConv(nn.Module):
    def __init__(self, cfg):
        super(PartSegHeadDeconvConv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=cfg['in_c'], out_channels=cfg['mid_c'], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cfg['mid_c'])
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=cfg['mid_c'], out_channels=cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.deconv.weight, std=0.001)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(self.deconv(x))))
        return x


class PartSegHeadDeconvDeconvConv(nn.Module):
    def __init__(self, cfg):
        super(PartSegHeadDeconvDeconvConv, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=cfg['in_c'], out_channels=cfg['mid_c'], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg['mid_c'])
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels=cfg['mid_c'], out_channels=cfg['mid_c'], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg['mid_c'])
        self.relu2 = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=cfg['mid_c'], out_channels=cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.deconv1.weight, std=0.001)
        nn.init.normal_(self.deconv2.weight, std=0.001)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(self.relu2(self.bn2(self.deconv2(self.relu1(self.bn1(self.deconv1(x)))))))
        return x