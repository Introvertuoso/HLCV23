import argparse
import json
import os
import time

import requests
import torch
import tarfile
from models import clip
import dataloaders

from urllib.request import urlopen
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from utils import extract_ds_features, knn_classifier

model_choices = [
    'clip',
]
dataset_choices = [
    'tiny',
]
corruption_choices = [
    'brightness',
    'contrast',
    'defocus_blur',
    'elastic_transform',
    'fog',
    'frost',
    'gaussian_noise',
    'glass_blur',
    'impulse_noise',
    'motion_blur',
    'pixelate',
    'shot_noise',
    'snow',
    'zoom_blur',
    'jpeg_compression',
]

def validate_argument(arg: list, options: list):
    to_remove = []
    for i in arg:
        if i not in options:
            to_remove.append(i)
            print(f'{i} not supported and will be skipped.')
    for i in to_remove:
        arg.remove(i)
    return arg

def main(args):
    # Adapted from https://github.com/sail-sg/MMCBench/tree/main

    # Check arguments
    model_list = validate_argument(args.model, model_choices)
    dataset_list = validate_argument(args.dataset, dataset_choices)
    corruption_list = validate_argument(args.corruption, corruption_choices)

    for mdl in model_list:
        # Set results up
        res = {
            'torch_version': torch.__version__,
            'model': '',
            'config': '',
            'dataset': '',
            'corruption': '',
            'device': args.device,
            'severity1': {},
            'severity2': {},
            'severity3': {},
            'severity4': {},
            'severity5': {},
            'time_elapsed': 0
        }

        # Load model
        if mdl == 'clip':
            model, config, transform, get_image_features = clip.define_model(device=args.device)
            res['model'] = mdl
            res['config'] = config

        for ds in dataset_list:
            # Load data
            path = os.path.join('../data', ds)
            c_path = path + '-c'
            if ds == 'tiny':
                if os.path.exists(path):
                    os.makedirs(path)
                    response = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip', stream=True)

                    total_size = int(response.headers.get("content-length", 0))
                    block_size = 1024

                    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                        with open(os.path.join(path, 'temp.file'), "wb") as file:
                            for data in response.iter_content(block_size):
                                progress_bar.update(len(data))
                                file.write(data)

                    if total_size != 0 and progress_bar.n != total_size:
                        raise RuntimeError("Could not download file")

                    ZipFile(os.path.join(path, 'temp.file'), 'r').extractall(path=path)
                    os.remove(os.path.join(path, 'temp.file'))

                if os.path.exists(c_path):
                    os.makedirs(c_path)
                    response = requests.get('https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar?download=1', stream=True)

                    total_size = int(response.headers.get("content-length", 0))
                    block_size = 1024

                    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                        with open(os.path.join(c_path, 'temp.file'), "wb") as file:
                            for data in response.iter_content(block_size):
                                progress_bar.update(len(data))
                                file.write(data)

                    if total_size != 0 and progress_bar.n != total_size:
                        raise RuntimeError("Could not download file")

                    tarfile.open(os.path.join(c_path, 'temp.file'), mode="r|*").extractall(path=c_path)
                    os.remove(os.path.join(c_path, 'temp.file'))
                res['dataset'] = ds

                for cor in corruption_list:
                    # Set corruption
                    res['corruption'] = cor
                    directory = os.path.join('../results/', mdl, ds, cor)
                    os.makedirs(directory, exist_ok=True)

                    start = time.time()
                    for sev in tqdm([1, 2, 3, 4, 5]):
                        # Compute KNN classifier for each severity
                        corrupt, num_classes = dataloaders.tiny_imagenet.corrupt('../', corruption_name=cor, severity=sev, transform=transform)
                        features, labels = extract_ds_features(model, corrupt, get_image_features, args.device)
                        res['severity'+str(sev)] = knn_classifier(features, labels, features, labels, num_classes=num_classes)
                        res['time_elapsed'] = time.time() - start

                        # Save results
                        with open(os.path.join(directory, time.asctime())+'.json', "w") as outfile:
                            json.dump(res, outfile)

                    # Print final results
                    print(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for evaluating model performance(s) on a given dataset(s) and corruption type(s).")
    parser.add_argument('--model', type=str, choices=model_choices, action='extend', nargs='+', required=True, help="Model name(s)")
    parser.add_argument('--dataset', type=str, choices=dataset_choices, action='extend', nargs='+', required=True, help="Dataset name(s)")
    parser.add_argument('--corruption', type=str, choices=corruption_choices, action='extend', nargs='+', default=['jpeg_compression'], help="Image corruption type(s)")
    parser.add_argument('--device', type=str, default='cpu', help="Computation device")
    args = parser.parse_args()

    main(args)
