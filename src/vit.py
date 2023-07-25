#pip install torchvision timm
import requests
from PIL import Image
from torchvision import transforms
import torch
import timm

# Define the Vision Transformer model
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Transform input for the model
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_features(image):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        features = model.forward_features(input_batch)

    return features

if __name__ == '__main__':
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    features = get_image_features(image)
    
    print(features.shape)
    torch.Size([1, 197, 768])
