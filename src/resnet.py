import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision.transforms import transforms


def define_model(device='cuda'):
    resnet152 = models.resnet152(pretrained=True).eval()
    modules = list(resnet152.children())[:-1]
    resnet152 = nn.Sequential(*modules).eval()
    for p in resnet152.parameters():
        p.requires_grad = False

    return resnet152.to(device)


def get_transform():
    return transforms.Compose(
        [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


feature_dim = 2048


def get_image_features(resnet_model, img_tensor):
    with torch.no_grad():
        features = resnet_model(img_tensor)

    return features[:, :, 0, 0]

# img = torch.randn(1, 3, 224, 224)

# features = resnet152(img)
