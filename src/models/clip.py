import torch
import PIL.Image as Image
import requests
from functools import partial
from torch import nn

import clip
from src.models.base import BaseModel

class CLIPModel(BaseModel):
    def __init__(self, backbone="ViT-B/16", device='cuda'):
        super().__init__(feature_dim=512, device=device)
        self.model, self.processor = clip.load(backbone, device=device)
        self.preprocess_fn = self.processor
        # #TODO: make model configs from a yaml file
    
    def preprocess(self, image):
        ...
    
    def forward(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
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
