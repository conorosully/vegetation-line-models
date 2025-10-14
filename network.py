
import torch
import torch.nn as nn
import torch.nn.functional as F

class SideOutput(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # 1 output channel as binary mask

    def forward(self, x, target_size):
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class HED(nn.Module):
    def __init__(self, backbone, out_channels=1, in_channels=4, input_size=(144, 144)):
        super().__init__()
        self.backbone = backbone


        # Do a dry forward pass to detect channel sizes
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, *input_size)
            features = backbone(dummy_input)
            self.side_channels = [f.shape[1] for f in features]

        # Create side output blocks with correct input channels
        self.side1 = SideOutput(in_channels=self.side_channels[0], out_channels=out_channels)
        self.side2 = SideOutput(in_channels=self.side_channels[1], out_channels=out_channels)
        self.side3 = SideOutput(in_channels=self.side_channels[2], out_channels=out_channels)
        self.side4 = SideOutput(in_channels=self.side_channels[3], out_channels=out_channels)
        self.side5 = SideOutput(in_channels=self.side_channels[4], out_channels=out_channels)

        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2:]

        c1, c2, c3, c4, c5 = self.backbone(x)

        s1 = self.side1(c1, (H, W))
        s2 = self.side2(c2, (H, W))
        s3 = self.side3(c3, (H, W))
        s4 = self.side4(c4, (H, W))
        s5 = self.side5(c5, (H, W))

        fused = self.fuse(torch.cat([s1, s2, s3, s4, s5], dim=1))

        return [s1, s2, s3, s4, s5, fused]
    
class SimpleCNNBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleCNNBackbone, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.stage1(x)   # [B, 64, H, W]
        out2 = self.stage2(out1)  # [B, 128, H/2, W/2]
        out3 = self.stage3(out2)  # [B, 256, H/4, W/4]
        out4 = self.stage4(out3)  # [B, 512, H/8, W/8]
        out5 = self.stage5(out4)  # [B, 512, H/16, W/16]
        return [out1, out2, out3, out4, out5]
