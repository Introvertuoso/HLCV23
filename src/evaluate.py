import argparse
import json
import os
import time

import torch
from dataloaders import tiny_imagenet

from tqdm import tqdm
from utils import extract_ds_features, knn_classifier, get_model, download_and_extract, get_classifier, \
    train_classifier, evaluate
from torchvision import transforms

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


# d['Speckle Noise'] = speckle_noise
# d['Gaussian Blur'] = gaussian_blur
# d['Spatter'] = spatter
# d['Saturate'] = saturate

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
            'train_logs': {},
            'severity0': {},
            'severity1': {},
            'severity2': {},
            'severity3': {},
            'severity4': {},
            'severity5': {},
            'time_elapsed': 0
        }

        # Load model
        model = get_model(mdl, device=args.device)
        res['model'] = mdl
        # res['config'] = config[mdl.upper()]

        for ds in dataset_list:
            # Load data
            path = os.path.join('..', 'data', ds)
            c_path = path + '-c'
            if ds == 'tiny':
                if not os.path.exists(path):
                    tiny_url = 'https://drive.google.com/file/d/1x5TptuPTwiXTbyX0XrltUQ6XAmCv9i5H/view?usp=share_link'
                    download_and_extract(path, tiny_url)

                if not os.path.exists(c_path):
                    tiny_c_url = 'https://drive.google.com/file/d/1p1XvarMzwmxEbR1qf9H04LsBNSANdNHE/view?usp=sharing'
                    download_and_extract(c_path, tiny_c_url)

                res['dataset'] = ds

                train_loader, num_classes = tiny_imagenet.clean('..', transform=model.preprocess_fn, split='train', num_workers=4)
                val_loader, _ = tiny_imagenet.clean('..', transform=model.preprocess_fn, num_workers=4)
                clf = get_classifier(model.feature_dim, num_classes)
                res['train_logs'] = train_classifier(clf, train_loader, val_loader, model, device=args.device)

                res['severity0'] = {'accuracy': res['train_logs']['val_accuracies'][-1]}

                for cor in corruption_list:  # where is the normal dataset?
                    # Set corruption
                    res['corruption'] = cor
                    directory = os.path.join('..', 'results', mdl, ds, cor)
                    os.makedirs(directory, exist_ok=True)

                    start = time.time()
                    for sev in tqdm([1, 2, 3, 4, 5]):
                        # Compute KNN classifier for each severity
                        corrupt, num_classes = tiny_imagenet.corrupt('..', corruption_name=cor, severity=sev,
                                                                     transform=model.preprocess_fn)  # only add resize transform here
                        # features, labels = extract_ds_features(corrupt, model, args.device)
                        # res['severity' + str(sev)] = knn_classifier(features, labels, features, labels, num_classes=num_classes)
                        res['severity' + str(sev)] = {'accuracy': evaluate(clf, model, corrupt, device=args.device)[1]}

                        res['time'] = time.time() - start
                        # Save results
                        with open(os.path.join(directory, time.asctime()) + '.json', "w") as outfile:
                            json.dump(res, outfile)

                    # Print final results
                    print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for evaluating model performance(s) on a given dataset(s) and corruption type(s).")
    parser.add_argument('--model', type=str, choices=model_choices, action='extend', nargs='+', required=True,
                        help="Model name(s)")
    parser.add_argument('--dataset', type=str, choices=dataset_choices, action='extend', nargs='+', required=True,
                        help="Dataset name(s)")
    parser.add_argument('--corruption', type=str, choices=corruption_choices, action='extend', nargs='+',
                        default=['brightness'], help="Image corruption type(s)")
    parser.add_argument('--device', type=str, default='cpu', help="Computation device")
    ## add config file
    # parser.add('--config_file', type=str, default='config.yaml', help='config file for the experiment')
    args = parser.parse_args()

    main(args)
