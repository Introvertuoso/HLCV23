# The code here was taken from the huggingface docs:
# https://huggingface.co/docs/transformers/model_doc/blip#transformers.BlipImageProcessor

import PIL.Image as Image
import requests
import torch
from transformers import AutoProcessor, BlipModel
from functools import partial

# model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
@torch.no_grad()
def get_image_features(model, processor, img_tensor, device='cuda'):
    inputs = processor(images=img_tensor, return_tensors="pt").to(device)
    image_features = model.get_image_features(**inputs)
    return image_features

def define_model(backbone="Salesforce/blip-image-captioning-base", device='cuda'):
    model = BlipModel.from_pretrained(backbone).to(device)
    processor = AutoProcessor.from_pretrained(backbone)
    get_image_features_fn = partial(get_image_features, model, processor)
    return model, get_image_features_fn




if __name__ == '__main__':
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    print(image.size)
    features = get_image_features(image)

    print(features.shape)
