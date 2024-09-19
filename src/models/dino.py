import torch
import PIL.Image as Image
import requests
from functools import partial
from torch import nn 
from models.base import BaseModel
from torchvision import transforms, datasets

feature_dim = 768

class DINOModel(BaseModel):
    def __init__(self, backbone="dino_vitb16", device='cuda'):
        super().__init__(feature_dim=feature_dim, device=device)
        self.model = torch.hub.load('facebookresearch/dino:main', backbone).to(self.device) #TODO: to change to v2 --> dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.transform = self.get_transform()
        self.preprocess_fn = self.transform
        #TODO: make model configs from a yaml file 
    
    def preprocess(self, image):
        return self.transform(image)
    
    def get_transform(self):

        return transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    def forward(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            image_features = self.model(img_tensor)
        return image_features.float()
        
        
# @torch.no_grad()
# def get_image_features(model, processor, img_tensor):
#     inputs = processor(img_tensor)
#     with torch.no_grad():
#         image_features = model.encode_image(inputs)
#     return image_features.float()


# def define_model(backbone="ViT-B/16", device='cuda'):
    
#     model, preprocess = clip.load(backbone, device=device)
#     model = model.eval().to(device)
#     get_image_features_fn = partial(get_image_features, model, preprocess)
#     return model, get_image_features_fn 


# def get_image_features(model, img_tensor):
#     inputs = processor(images=img_tensor, return_tensors="pt")
#     image_features = model.get_image_features(**inputs)
#     return image_features




if __name__ == '__main__':
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    print(image.size)
    features = get_image_features(image)

    print(features.shape)
