import argparse
import os
import torch
import tarfile
import models
import dataloaders

from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def main(args):
    # Adapted from https://github.com/sail-sg/MMCBench/tree/main

    for mdl in args.model:
        # Load model
        if mdl == 'clip':
            model, transform = models.clip.define_model(device=args.device)
        else:
            print(f'Model {args.model} not supported')
            continue

        for ds in args.dataset:
            # Load data
            path = os.path.join('../data', ds)
            if not os.path.exists(path):
                os.makedirs(os.path.join(path))
            if ds == 'tiny':
                http_response = urlopen('http://cs231n.stanford.edu/tiny-imagenet-200.zip')
                ZipFile(BytesIO(http_response.read())).extractall(path=path)
                clean = dataloaders.tiny_imagenet.clean()

                # Train classifier on clean data

                http_response = urlopen('https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar?download=1')
                tarfile.open(http_response, mode="r|gz").extractall(path=os.path.join(path, '-c'))
                for cor in args.corruption:
                    # Set corruption
                    for sev in [1, 2, 3, 4, 5]:
                        # Set severity
                        corrupt = dataloaders.tiny_imagenet.corrupt(corruption=cor, severity=sev)

                        # Test on corrupted data

                    # Save results (train + test)
                    directory = f"results/{args.model}"
                    os.makedirs(directory, exist_ok=True)

                    # Print  results (train + test)


            else:
                print(f'Dataset {args.dataset} not supported')
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for evaluating model performance on a given dataset and corruption type.")
    parser.add_argument('--model', type=str, choices=['clip'], action='extend', nargs='+', required=True,
                        help="Model name")
    parser.add_argument('--dataset', type=str, choices=['tiny'], action='extend', nargs='+', required=True,
                        help="Dataset name")
    parser.add_argument('--corruption', type=str, choices=[
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
        'jpeg_compression'
    ], action='extend', nargs='+', default=['jpeg_compression'], help="Image corruption type")
    parser.add_argument('--device', type=str, default='cpu', help="Computation device")
    args = parser.parse_args()

    main(args)
