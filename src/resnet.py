import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
resnet152 = models.resnet152(pretrained=True)
modules=list(resnet152.children())[:-1]

resnet152 = nn.Sequential(*modules)
for p in resnet152.parameters():
    p.requires_grad = False
    

def get_image_features(img_tensor):
    features = resnet152(img_tensor)
    
    return features

# img = torch.randn(1, 3, 224, 224)

# features = resnet152(img) 

