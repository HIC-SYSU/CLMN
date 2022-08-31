import torch
import torch.nn as nn

class BTNK1(nn.Module):
    def __init__(self, in_dims=32, mid_dims=64, s=2):
        super(BTNK1, self).__init__()
        factor = 4
        self.mp = nn.MaxPool2d(2, stride=2)
        self.sp_block1 = nn.Sequential(
            nn.Conv2d(in_dims, mid_dims, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dims, mid_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dims, mid_dims * factor, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_dims * factor)
        )

        self.sp_block2 = nn.Sequential(
            nn.Conv2d(in_dims, mid_dims * factor, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_dims * factor)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.mp(x)
        x1 = self.sp_block1(x)
        x2 = self.sp_block2(x)
        x = self.relu(x1 + x2)
        return x


class BTNK2(nn.Module):
    def __init__(self, in_dims=32):
        super(BTNK2, self).__init__()

        factor = 4
        self.sp_block1 = nn.Sequential(
            nn.Conv2d(in_dims, in_dims // factor, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dims // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims // factor, in_dims // factor, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dims // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims // factor, in_dims, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dims)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.sp_block1(x)
        x = self.relu(x1 + x)
        return x


class SuperPixelNet(nn.Module):
    def __init__(self, feature_root=32, grid_size=14):
        super(SuperPixelNet, self).__init__()
        self.grid_size = grid_size

        self.sp_block1 = nn.Sequential(
            nn.Conv2d(3, feature_root // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_root // 2, feature_root // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root // 2),
            nn.ReLU(inplace=True)
        )

        self.sp_block2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(feature_root // 2, 2 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * feature_root, 2 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * feature_root),
            nn.ReLU(inplace=True)
        )

        self.sp_stage1 = nn.Sequential(
            BTNK1(in_dims=64, mid_dims=64, s=1),
            BTNK2(in_dims=256),
            BTNK2(in_dims=256)
        )

        self.sp_stage2 = nn.Sequential(
            BTNK1(in_dims=256, mid_dims=128, s=2),
            BTNK2(in_dims=512),
            BTNK2(in_dims=512),
            BTNK2(in_dims=512),
        )

        self.sp_stage3 = nn.Sequential(
            BTNK1(in_dims=512, mid_dims=256, s=2),
            BTNK2(in_dims=1024),
            BTNK2(in_dims=1024),
            BTNK2(in_dims=1024),
            BTNK2(in_dims=1024),
            BTNK2(in_dims=1024)
        )

        self.sp_stage4 = nn.Sequential(
            BTNK1(in_dims=1024, mid_dims=512, s=2),
            BTNK2(in_dims=2048),
            BTNK2(in_dims=2048)
        )

        self.SP_up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.SP_deconvBlock1 = nn.Sequential(
            nn.Conv2d(32 * feature_root, 8 * feature_root, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(8 * feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * feature_root, 8 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * feature_root),
            nn.ReLU(inplace=True)
        )

        self.SP_up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.SP_deconvBlock2 = nn.Sequential(
            nn.Conv2d(16 * feature_root, 4 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * feature_root, 4 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * feature_root),
            nn.ReLU(inplace=True)
        )

        self.SP_up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.SP_deconvBlock3 = nn.Sequential(
            nn.Conv2d(8 * feature_root, 4 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * feature_root, 2 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * feature_root),
            nn.ReLU(inplace=True)
        )

        self.SP_up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.SP_deconvBlock4 = nn.Sequential(
            nn.Conv2d(4 * feature_root, 2 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * feature_root, feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root),
            nn.ReLU(inplace=True)
        )

        self.SP_up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.SP_deconvBlock5 = nn.Sequential(
            nn.Conv2d(feature_root + feature_root // 2, feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root),
            nn.ReLU(inplace=True)
        )

        self.proj_prob = nn.Sequential(
            nn.Conv2d(feature_root, 9, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

        self.sp_block3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.sp_block4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.sp_block5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.sp_block6 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        layer1 = self.sp_block1(x)
        layer2 = self.sp_block2(layer1)

        stage1 = self.sp_stage1(layer2)
        stage2 = self.sp_stage2(stage1)
        stage3 = self.sp_stage3(stage2)
        stage4 = self.sp_stage4(stage3)

        layer3 = self.sp_block3(stage1)
        layer4 = self.sp_block4(stage2)
        layer5 = self.sp_block5(stage3)
        layer6 = self.sp_block6(stage4)

        x = self.SP_up1(layer6)
        x = torch.cat([x, layer5], dim=1)
        x = self.SP_deconvBlock1(x)

        x = self.SP_up2(x)
        x = torch.cat([x, layer4], dim=1)
        x = self.SP_deconvBlock2(x)

        x = self.SP_up3(x)
        x = torch.cat([x, layer3], dim=1)
        x3 = self.SP_deconvBlock3(x)

        x = self.SP_up4(x3)
        x = torch.cat([x, layer2], dim=1)
        x = self.SP_deconvBlock4(x)

        x = self.SP_up5(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.SP_deconvBlock5(x)

        prob = self.proj_prob(x)
        return prob, layer2, stage1
