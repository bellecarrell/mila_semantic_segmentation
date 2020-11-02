from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import deeplabv3_resnet50


def get_fcn():
    return fcn_resnet50(pretrained=False, num_classes=2)


def get_deeplab():
    return deeplabv3_resnet50(pretrained=True, num_classes=1)
