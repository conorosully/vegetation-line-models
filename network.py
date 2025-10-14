
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

import warnings

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
    
def get_ResNet50_BigEarthNet(in_channels=4):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        backbone = BigEarthNetv2_0_ImageClassifier.from_pretrained(
        "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
        )
    
    backbone = backbone.model.vision_encoder

    # Replace the first convolutional layer
    old_conv = backbone.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False  # BigEarthNet models use bias=False
    )

    with torch.no_grad():
        # ResNet conv1 expects [R, G, B] channel order
        # We'll map NIR to the average of RGB filters (or initialize it smartly)
        #see https://huggingface.co/BIFOLD-BigEarthNetv2-0 for band order
        new_conv.weight[:, 0] = old_conv.weight[:, 0]  # Blue (B2)
        new_conv.weight[:, 1] = old_conv.weight[:, 1]  # Green (B3)
        new_conv.weight[:, 2] = old_conv.weight[:, 2]  # Red (B4) 
        new_conv.weight[:, 3] = old_conv.weight[:, 6]  # NIR (B8)
        if in_channels == 5:
            # We need a channel for guidance band
            new_conv.weight[:, 4] = old_conv.weight.mean(dim=1)


    backbone.conv1 = new_conv

    return_nodes = {
        'act1': 'out1',           # after initial conv + BN + ReLU
        'layer1': 'out2',         # low-level features
        'layer2': 'out3',
        'layer3': 'out4',
        'layer4': 'out5'          # deepest features
    }

    return return_nodes, backbone

    

def get_ResNet50_ImageNet(in_channels=4):
    """
    Load a pretrained ResNet-50 model and modify the first conv layer 
    to accept 4 channels (BGR + NIR).
    """
    
    # Load pretrained ResNet-50
    weights = ResNet50_Weights.DEFAULT
    backbone = resnet50(weights=weights)

    # Modify the first conv layer if needed


    # Replace the first convolutional layer
    old_conv = backbone.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        # ResNet conv1 expects [R, G, B] channel order
        # We'll map NIR to the average of RGB filters (or initialize it smartly)
        new_conv.weight[:, 0] = old_conv.weight[:, 2]  # Blue  <- ResNet Blue (channel 2)
        new_conv.weight[:, 1] = old_conv.weight[:, 1]  # Green <- ResNet Green (channel 1)
        new_conv.weight[:, 2] = old_conv.weight[:, 0]  # Red   <- ResNet Red (channel 0)
        new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)  # NIR  <- Avg(RGB) as a proxy
        if in_channels == 5:
            new_conv.weight[:, 4] = old_conv.weight.mean(dim=1)

    backbone.conv1 = new_conv

    # Define return nodes
    return_nodes = {
        'relu': 'out1',           # after initial conv + BN + ReLU
        'layer1': 'out2',         # low-level features
        'layer2': 'out3',
        'layer3': 'out4',
        'layer4': 'out5'          # deepest features
    }

    return return_nodes, backbone
    

class ResNet50Backbone(nn.Module):
    def __init__(self, in_channels=3,backbone_dataset='ImageNet', freeze_backbone=False):
        super(ResNet50Backbone, self).__init__()

        if backbone_dataset == 'ImageNet':
             return_nodes, backbone = get_ResNet50_ImageNet(in_channels)
        elif backbone_dataset == 'BigEarthNet':
            return_nodes, backbone = get_ResNet50_BigEarthNet(in_channels)
             
         # Optionally freeze backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            # Unfreeze the first conv layer as it has been modified
            for param in backbone.conv1.parameters():
                param.requires_grad = True
  
        # Create extractor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)

    def forward(self, x):
        features = self.backbone(x)
        return [features['out1'], features['out2'], features['out3'],
                features['out4'], features['out5']]

class HEDModel(nn.Module):
    def __init__(self, version="simplecnn", in_channels=5, out_channels=1):
        super().__init__()
        if version == "simplecnn":
            backbone = SimpleCNNBackbone(in_channels)
        elif version == "bigearthnet":
            backbone = ResNet50Backbone(in_channels)
        else:
            raise ValueError("version must be 'simplecnn' or 'bigearthnet'")
        self.model = HED(backbone, in_channels, out_channels)

    @classmethod
    def from_pretrained(cls, repo_id, version="simplecnn", device=None):
        from huggingface_hub import hf_hub_download
        model = cls(version)
        filename = "SIVE_SimpleCNN.pth" if version == "simplecnn" else "SIVE_BigEarthNet.pth"
        weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model.model.load_state_dict(torch.load(weight_path, map_location=device or "cpu"))
        model.eval()
        return model