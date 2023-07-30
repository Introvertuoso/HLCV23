import torch
from tqdm import tqdm
import torch.nn.functional as F


@torch.no_grad()
def extract_ds_features(model, data_loader, get_features_fn, device):
    """
    pass the torch model and the dataloader along with the get_img_features function
    """
    feature_list, labels_list = [], []
    for batch in tqdm(data_loader, leave=False):
        img_tensor, labels = batch[0].to(device), batch[1].to(device)
        feature_tensor = get_features_fn(model, img_tensor)
        feature_list.append(feature_tensor)
        labels_list.append(labels)

    all_features_tensor = torch.cat(feature_list)
    all_labels_tensor = torch.cat(labels_list)

    return all_features_tensor, all_labels_tensor


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
    return top1, top5
