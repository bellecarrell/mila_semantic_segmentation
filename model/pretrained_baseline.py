from torchvision.models import resnet50


def get_baseline():
    return resnet50(pretrained=True)
