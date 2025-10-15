dependencies = ['torch', 'torchvision', 'huggingface_hub']
import network
import torch
import os

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

    # Build the full path to the weights in the repo folder
    repo_dir = os.path.dirname(__file__)
    weights_path = os.path.join(repo_dir, state_dict)

    # Load weights
    state_dict_loaded = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict_loaded)
    model.eval()

    meta_data = {
        'name': "SIVE_SimpleCNN",
        'date': "04/JUNE/2024",
        'arcitecture': "HED",
        'backbone': "SimpleCNN",
        'freeze_backbone': False,
        'guidance': True,
        'loss_function': "wBCE"
    }

    return model, meta_data