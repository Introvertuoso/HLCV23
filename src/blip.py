# The code here was taken from the huggingface docs:
# https://huggingface.co/docs/transformers/model_doc/blip#transformers.BlipImageProcessor

import PIL.Image as Image
import requests
from transformers import AutoProcessor, BlipModel

model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")


def get_image_features(img_tensor):
    inputs = processor(images=img_tensor, return_tensors="pt").to('cuda:0')
    model.to('cuda:0')
    image_features = model.get_image_features(**inputs)
    return image_features


if __name__ == '__main__':
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    print(image.size)
    features = get_image_features(image)

    print(features.shape)
