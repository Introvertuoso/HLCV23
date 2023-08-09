# %%
import torch
import sys 
sys.path.append('..')
from src.dataloaders import imagenet_c_dataloader, imagenet_dataloader
from tqdm import tqdm
from src.imagebind import define_model, get_image_features, get_transform
from src.knn_clf_score import extract_ds_features, knn_classifier

# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# %%
model = define_model(device)
transform = get_transform()

# # %%
gaussian_noise_acc = []
for sev in tqdm([1, 2, 3, 4, 5]):
    loader = imagenet_c_dataloader(corruption_name='gaussian_noise', severity=sev, batch_size=256, transform=transform)
    features, labels = extract_ds_features(model, loader, get_image_features, device)
    gaussian_noise_acc.append([knn_classifier(features, labels, features, labels, num_classes=1000)])

# # %%
print("gaussian_noise_acc", gaussian_noise_acc )

# %%
impulse_noise_acc = []
for sev in tqdm([1, 2, 3, 4, 5]):
    loader = imagenet_c_dataloader(corruption_name='impulse_noise', severity=sev, batch_size=256, transform=transform)
    features, labels = extract_ds_features(model, loader, get_image_features, device)
    impulse_noise_acc.append([knn_classifier(features, labels, features, labels, num_classes=1000)])

# %%
print("impulse_noise_acc", impulse_noise_acc)

# %%
shot_noise_acc = []
for sev in tqdm([1, 2, 3, 4, 5]):
    loader = imagenet_c_dataloader(corruption_name='shot_noise', severity=sev, batch_size=256, transform=transform)
    features, labels = extract_ds_features(model, loader, get_image_features, device)
    shot_noise_acc.append([knn_classifier(features, labels, features, labels, num_classes=1000)])

# %%
print("shot_noise_acc", shot_noise_acc)

# # %%
speckle_noise_acc = []
for sev in tqdm([1, 2, 3, 4, 5]):
    loader = imagenet_c_dataloader(corruption_name='speckle_noise', severity=sev, batch_size=256, transform=transform)
    features, labels = extract_ds_features(model, loader, get_image_features, device)
    speckle_noise_acc.append([knn_classifier(features, labels, features, labels, num_classes=1000)])

# %%
print("speckle_noise_acc", speckle_noise_acc)

# # %%
loader = imagenet_dataloader(batch_size=256, transform=transform)
features, labels = extract_ds_features(model, loader, get_image_features, device)
clean_top1, clean_top5 = knn_classifier(features, labels, features, labels, num_classes=1000)

# %%
print('clean acc')
print(clean_top1, clean_top5)

# %%
res = {
    'gaussian_noise_acc' : gaussian_noise_acc,
    'impulse_noise_acc' : impulse_noise_acc,
    'shot_noise_acc': shot_noise_acc,
    'speckle_noise_acc': speckle_noise_acc,
    'clean_acc': [clean_top1, clean_top5]
}
print(res)

# %%


# %%



