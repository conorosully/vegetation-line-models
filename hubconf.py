dependencies = ['torch', 'torchvision', 'huggingface_hub']
from src.model import HEDModel

def hed_simplecnn(pretrained=True):
    model = HEDModel("simplecnn")
    if pretrained:
        model = HEDModel.from_pretrained("conorosully/HED-Coastline-Detection", version="simplecnn")
    return model

def hed_bigearthnet(pretrained=True):
    model = HEDModel("bigearthnet")
    if pretrained:
        model = HEDModel.from_pretrained("conorosully/HED-Coastline-Detection", version="bigearthnet")
    return model