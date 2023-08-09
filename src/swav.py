import torch
import torchvision.transforms as transforms

feature_dim = 2048


def define_model(device='cuda', ):
    model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    model = model.eval()
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules).eval()
    for p in model.parameters():
        p.requires_grad = False

    return model.to(device)


def preprocess(img):
    im_transform = get_transform()
    return im_transform(img)


def get_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def get_image_features(model, img_tensor):
    return model(img_tensor).squeeze()