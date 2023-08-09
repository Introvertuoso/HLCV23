# %%
import torch
import os

import numpy as np
import sys 
sys.path.append('..')
from imagebind import data
from imagebind.models import imagebind_model
from src.imagenet_labels import lab_dict
from tqdm import tqdm
from imagebind.models.imagebind_model import ModalityType

# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# %%
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model = model.to(device)

# %%
text_list = [lab_dict[i].replace('_', ' ') for i in os.listdir('../imagenet_data/imagenet')]
text_list = [f"a {c}" for c in text_list]

# %%
text_list

# %%
def get_acc(gt, preds = None):
    if preds is not None: 
        return ((preds.argmax(1)==gt).sum()/len(preds)).cpu().numpy()
    return ((preds.argmax(1)==gt).sum()/len(preds)).cpu().numpy()


def compute(model, text, images, labels, device):
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text, device),
        ModalityType.VISION: data.load_and_transform_vision_data(images, device),
    }
    
    with torch.no_grad():
        embeddings = model(inputs)
    
    probs = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)
    # print(probs, labels)
    val_acc = get_acc(labels.to(device), probs)
    return val_acc
    
def get_image_paths(root):
    path_dict = {}
    for cls in tqdm(os.listdir(root)):
        path_list = []
        cls_path = os.path.join(root, cls)
        for img in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img)
            path_list.append(img_path)
        path_dict[lab_dict[cls].replace('_', ' ')] = path_list
    return path_dict

def get_test_acc(image_paths, device):
    eval_acc = []
    for i in tqdm(range(len(text_list)//20)):
        image_paths_batch = []
        labels = []
        for j in range(i*20, (i+1)*20):
            image_paths_batch += (image_paths[text_list[j][2:]])
            labels += [j]*len((image_paths[text_list[j][2:]]))
        
        # print(image_paths_batch)
        # print(labels)
        eval_acc.append(
            compute(model, text_list, image_paths_batch, torch.tensor(labels), device)
        ) # 50 samples per class; first 2 chars are "a "
        
    return np.mean(eval_acc)


# %%
path_to_imagenet = '../imagenet_data/imagenet'
path_to_imagenet_c = '../imagenet_data/'

# %%
image_paths = get_image_paths(path_to_imagenet)


# %%
# clean_acc = get_test_acc(image_paths, device)

# %%
# clean_acc

# %%
gaussian_noise_acc = []
for sev in tqdm([1, 2, 3, 4, 5]):
    image_paths = get_image_paths(os.path.join(path_to_imagenet_c, 'gaussian_noise', str(sev)))
    gaussian_noise_acc.append(get_test_acc(image_paths, device))

# %%
print(gaussian_noise_acc)

# %%
impulse_noise_acc = []
for sev in tqdm([1, 2, 3, 4, 5]):
    image_paths = get_image_paths(os.path.join(path_to_imagenet_c, 'impulse_noise', str(sev)))
    impulse_noise_acc.append(get_test_acc(image_paths, device))

# %%
print(impulse_noise_acc)

# # %%
# shot_noise_acc = []
# for sev in tqdm([1, 2, 3, 4, 5]):
#     image_paths = get_image_paths(os.path.join(path_to_imagenet_c, 'shot_noise', str(sev)))
#     shot_noise_acc.append(get_test_acc(image_paths, device))

# # %%
# shot_noise_acc

# # %%
# speckle_noise_acc = []
# for sev in tqdm([1, 2, 3, 4, 5]):
#     image_paths = get_image_paths(os.path.join(path_to_imagenet_c, 'speckle_noise', str(sev)))
#     speckle_noise_acc.append(get_test_acc(image_paths, device))

# # %%
# speckle_noise_acc

# %%
res = {
    'gaussian_noise_acc' : gaussian_noise_acc,
    'impulse_noise_acc' : impulse_noise_acc,
    # 'shot_noise_acc': shot_noise_acc,
    # 'speckle_noise_acc': speckle_noise_acc,
    # 'clean_acc': clean_acc
}

# %%p
#
print(res)


