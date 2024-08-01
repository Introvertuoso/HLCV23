import logging
import os
from zipfile import ZipFile

import numpy as np
import requests
import torch

import torch.nn.functional as F
from torch import nn

from tqdm import tqdm


def download_and_extract(path, url):
    os.makedirs(path)
    tmp_name = 'temp.zip'

    import platform
    if platform.system() == 'Windows':
        import gdown
        gdown.download(url, os.path.join(path, tmp_name), quiet=False, fuzzy=True)

    else:
        response = requests.get(url, stream=True)

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(os.path.join(path, tmp_name), "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")

    ZipFile(os.path.join(path, tmp_name), 'r').extractall(path=path)
    os.remove(os.path.join(path, tmp_name))


def get_model(model_name, device='cuda', **kwargs):
    if model_name == 'clip':
        from models.clip import CLIPModel
        backbone = kwargs.get('backbone', 'ViT-B/32')
        model = CLIPModel(backbone, device=device)
        return model
    if model_name == 'blip':
        from models.blip import BLIPModel
        backbone = kwargs.get('backbone', 'Salesforce/blip-image-captioning-base')
        model = BLIPModel(backbone, device=device)
        return model
    # elif model_name == 'vit':
    #     from transformers import ViTFeatureExtractor, ViTForImageClassification
    #     model = ViTForImageClassification.from_pretrained("nateraw/vit-base-patch16-224-in21k")
    #     model = model.eval()
    #     model_config = "nateraw/vit-base-patch16-224-in21k"
    #     preprocess = ViTFeatureExtractor.from_pretrained("nateraw/vit-base-patch16-224-in21k")
    #     return model, model_config, preprocess, get_image_features_vit
    # else:
    #     raise ValueError("Model not supported"


@torch.no_grad()
def extract_ds_features(data_loader, model, device):
    """
    pass the torch model and the dataloader along with the get_img_features function
    """
    feature_list, labels_list = [], []
    for batch in tqdm(data_loader, leave=False):
        img_tensor, labels = batch[0].to(device), batch[1].to(device)
        feature_tensor = model(img_tensor)
        feature_list.append(feature_tensor)
        labels_list.append(labels)

    all_features_tensor = torch.cat(feature_list)
    all_labels_tensor = torch.cat(labels_list)

    return all_features_tensor, all_labels_tensor


@torch.no_grad()
def extract_features(tensor, model):
    return model(tensor)


def get_classifier(embedding_size: int, num_of_classes: int):
    return nn.Sequential(nn.Linear(embedding_size, num_of_classes))


def get_acc(gt, preds):
    return ((preds.argmax(1) == gt).sum() / len(preds)).cpu().numpy()


def cache_embeddings(path, loader, model, device='cpu'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    embeddings = None
    labels = None
    for i, batch in enumerate(tqdm(loader, leave=True), 0):
        im, lbl = batch[0].to(device), batch[1].to(device)
        features = extract_features(im, model)
        if i == 0:
            embeddings, labels = features, lbl
        else:
            embeddings = torch.cat([embeddings, features], dim=0)
            labels = torch.cat([labels, lbl], dim=0)

    torch.save({'embeddings': embeddings, 'labels': labels}, path)


def evaluate(model, val_loader, embedding_model=None, loss_fn=nn.CrossEntropyLoss(), device='cpu'):
    model = model.to(device)
    eval_acc = []
    eval_losses = []
    for eval_batch in tqdm(val_loader, leave=False):
        ims, labels = eval_batch
        ims, labels = ims.to(device), labels.to(device)
        if embedding_model is not None:
            embedding_model = embedding_model.to(device)
            features = extract_features(ims, embedding_model)
        else:
            features = ims
        preds = model(features)
        loss_val = loss_fn(preds, labels.view(-1, ))
        val_acc = get_acc(labels.view(-1, ), preds)

        eval_acc.append(val_acc)
        eval_losses.append(loss_val.item())

    return np.mean(eval_losses), np.mean(eval_acc)


def train_classifier(clf_model, train_loader, val_loader, embedding_model=None, loss_fn=nn.CrossEntropyLoss(), epochs=5,
                     device='cpu'):
    optim = torch.optim.Adam(clf_model.parameters(), lr=0.001)
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    for ep in tqdm(range(epochs)):
        run_loss = 0.
        ep_losses = []
        ep_accs = []
        if ep == 0:
            eval_loss, eval_acc = evaluate(model=clf_model, val_loader=val_loader, embedding_model=embedding_model, loss_fn=loss_fn,
                                           device=device)
            print(f'initial loss {eval_loss} and initial accuracy {eval_acc}')

        for i, batch in enumerate(tqdm(train_loader, leave=False), 0):
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            optim.zero_grad()
            # if embedding is not None:
            #     features = extract_features(imgs, embedding)
            # else:
            #     features = imgs
            preds = clf_model(features.float())
            loss = loss_fn(preds, labels.view(-1, ))

            loss.backward()
            optim.step()

            ep_losses.append(loss.item())
            ep_accs.append(get_acc(labels.view(-1, ), preds))

        ep_loss = np.mean(ep_losses)
        losses.append(ep_loss)

        ep_acc = np.mean(ep_accs)
        accs.append(ep_acc)

        eval_loss, eval_acc = evaluate(model=clf_model, val_loader=val_loader, embedding=embedding, loss_fn=loss_fn,
                                       device=device)
        val_losses.append(eval_loss)
        val_accs.append(eval_acc)
        print(f' train loss: {ep_loss}, val loss: {eval_loss}, Train accuracy {ep_acc}, val accuracy {eval_acc} ')

    return {'train_losses': losses, 'train_accuracies': accs, 'val_losses': val_losses, 'val_accuracies': val_accs}


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k=5, num_classes=10):
    """
    pass train features and labels to be same as test if we don't have a train set features
    returns:  top1, top5 accuracy
    """
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    # print(train_features.shape)
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(torch.int64).to(train_features.device)
    all_idxs = torch.arange(test_labels.shape[0])
    for idx in tqdm(range(0, num_test_images, imgs_per_chunk), leave=False):
        # get the features for test images
        si, ei = idx, min((idx + imgs_per_chunk), num_test_images)
        ixs = (torch.arange(si, ei))
        # mask = all_idxs[all_idxs!=ixs]
        index = torch.ones(all_idxs.shape[0], dtype=bool)
        index[ixs] = False
        selected_ixx = all_idxs[index]
        # print(selected_ixx)
        features = test_features[ixs]
        # print(features.shape)
        targets = test_labels[ixs]
        batch_size = targets.shape[0]
        # print(train_features[selected_ixx].shape)
        train_features_f = train_features
        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features_f)
        distances, indices = similarity.topk(k + 1, largest=True, sorted=True)
        distances, indices = distances[:, 1:], indices[:, 1:]
        # print(distances.shape, indices.shape)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices).to(torch.int64)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        # print(torch.mode(retrieved_neighbors).values, targets)
        # print(targets)
        # print(retrieval_one_hot.argmax(1))
        # print((torch.min(distances, dim=1)).values.shape)
        distances_transform = F.softmax(distances)
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        # print(probs)
        _, predictions = probs.sort(1, True)  # torch.mode(retrieved_neighbors).values#probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        # print(correct.narrow(1, 0, 1).sum().item())
        top1 = top1 + correct.narrow(1, 0,
                                     1).sum().item()  # (predictions==targets).sum().item() #correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return {'accuracy': {'top1': top1, 'top5': top5}, }
