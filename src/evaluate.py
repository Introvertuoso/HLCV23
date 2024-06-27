import argparse
import json
import os
import time

import torch
import tarfile
import models
import dataloaders

from urllib.request import urlopen
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from utils import extract_ds_features, knn_classifier, get_model
from torchvision import transforms
import yaml 

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
        # if mdl == 'clip':
        model, get_image_features_fn  = get_model(mdl, device=args.device)
        res['model'] = mdl
        res['get_features_fn'] = get_image_features_fn
        
        res['config'] = config[mdl.upper()]

        for ds in dataset_list:
            # Load data
            path = os.path.join('../data', ds)
            c_path = os.path.join(path, '-c')
            if ds == 'tiny':
                if not os.path.exists(path):
                    os.makedirs(os.path.join(path))
                    http_response = urlopen('http://cs231n.stanford.edu/tiny-imagenet-200.zip')
                    ZipFile(BytesIO(http_response.read())).extractall(path=path)
                if not os.path.exists(c_path):
                    http_response = urlopen('https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar?download=1')
                    tarfile.open(http_response, mode="r|gz").extractall(path=c_path)
                res['dataset'] = ds

                for cor in corruption_list: # where is the normal dataset?
                    # Set corruption
                    res['corruption'] = cor
                    directory = os.path.join('../results/', mdl, ds, cor)
                    os.makedirs(directory, exist_ok=True)

                    start = time.time()
                    for sev in tqdm([1, 2, 3, 4, 5]):
                        # Compute KNN classifier for each severity
                        corrupt, num_classes = dataloaders.tiny_imagenet.corrupt('../', corruption_name=cor, severity=sev, transform=transforms.Resize((224, 224))) # only add resize transform here
                        features, labels = extract_ds_features(corrupt, get_image_features_fn, args.device)
                        res['severity'+str(sev)] = knn_classifier(features, labels, features, labels, num_classes=num_classes)
                        res['time'] = time.time() - start
                        # Save results
                        with open(os.path.join(directory, time.asctime())+'.json', "x") as outfile:
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
    ## add config file
    parser.add('--config_file', type=str, default='config.yaml', help='config file for the experiment')
    args = parser.parse_args()

    main(args)
