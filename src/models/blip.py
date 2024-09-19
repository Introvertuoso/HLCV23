# The code here was taken from the huggingface docs:
# https://huggingface.co/docs/transformers/model_doc/blip#transformers.BlipImageProcessor

import PIL.Image as Image
import requests
import torch
from lavis.models import load_model_and_preprocess
from models.base import BaseModel

class BLIPModel(BaseModel):
    def __init__(self, name="blip_feature_extractor", model_type="base", device='cuda'):
        super().__init__(feature_dim=512, device=device)
        self.model, self.processor, _ = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=True,
            device=self.device
        )
        self.preprocess_fn = self.processor["eval"].transform
        #TODO: make model configs from a yaml file 

    def preprocess(self, image):
        ...
    
    def forward(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        sample = {"image": img_tensor, "text_input": None}
        with torch.no_grad():
            image_features = self.model.extract_features(sample, mode="image").image_embeds
        return image_features.float()
        

# @torch.no_grad()
# def get_image_features(model, processor, img_tensor, device='cuda'):
#     inputs = processor(images=img_tensor, return_tensors="pt").to(device)
#     image_features = model.get_image_features(**inputs)
#     return image_features
#
# def define_model(backbone="Salesforce/blip-image-captioning-base", device='cuda'):
#     model = BlipModel.from_pretrained(backbone).to(device)
#     processor = AutoProcessor.from_pretrained(backbone)
#     get_image_features_fn = partial(get_image_features, model, processor)
#     return model, get_image_features_fn




if __name__ == '__main__':
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    print(image.size)
    # features = get_image_features(image)

    # print(features.shape)
