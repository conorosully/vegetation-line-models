dependencies = ['torch', 'torchvision', 'huggingface_hub']
import network
import torch

def get_model(state_dict,guidance=True):

    if guidance == True:
        in_channels = 5  # 4 input channels + 1 guidance channel
    else:
        in_channels = 4

    # Load model arcitecture
    backbone = network.SimpleCNNBackbone(in_channels=in_channels)
    model = network.HED(backbone=backbone, 
                    in_channels=in_channels,
                    out_channels=1)

    # Load weights
    state_dict = torch.load(state_dict, map_location=torch.device('cpu') )
    model.load_state_dict(state_dict)
    model.eval()

    return model