# https://github.com/facebookresearch/ImageBind

from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import torch
from torchvision.transforms import transforms

def get_image_features(model, img_tensor, device='cuda'):
    # image = preprocess(raw_image, device)
    # model = define_model(device)
    with torch.no_grad():
        return model({ModalityType.VISION: img_tensor.to(device)})[ModalityType.VISION]


def define_model(device):
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    return model


def preprocess(raw_image, device):
    data_transform = get_transform()
    return torch.stack([data_transform(raw_image).to(device)], dim=0)


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
