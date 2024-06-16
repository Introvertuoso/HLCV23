# Noise & Modalities

## Overview
To be added later.

## Benchmarking process
To be added later

### File Structure

- `configs` contains any model parameters used in the experiments.
- `data` contains all the necessary datasets.
- `docs` contains the files for serving the Sphinx documentation.
- `notebooks` contains the experiments where different methods/parameters are tested/applied.
- `packages` contains any third-party implementations.
- `references` contains any reference material (e.g papers, notes, slide decks, books/chapters, etc.).
- `results` contains any reports, tables, plots, or outputs organized in folders corresponding to experiment notebooks.
- `src` is where all the source code lives.
- `keynote` contains all the keynote files.
- `tests` contains any code testing routines.
- `writeup` contains all the writeup files.

## Installation

- Clone this repository. Name the folder `noise-modalities`.
- Create a virtual environment (call it `.venv` for consistency), activate it, and install the requirements.


    cd noise-modalities
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Usage

### Evaluation script

Use `evaluate.py` with the following arguments.

#### Command format

```
python evaluation.py --model [MODEL] --dataset [DATASET] --corruption [CORRUPTION] --device [DEVICE]
```
- `--model`: Model name (choices: `clip`).
- `--dataset`: Dataset name (choices: `tiny`, `tiny-c`).
- `--corruption`: Image corruption type. (choices: `brightness`, `contrast`, `defocus_blur`, `elastic_transform`, `fog`, `frost`, `gaussian_noise`, `glass_blur`, `impulse_noise`, `motion_blur`, `pixelate`, `shot_noise`, `snow`, `zoom_blur`, default= `jpeg_compression`). Applicable only to corrupted datasets (ending with `-c`).
- `--device`: Computation device (default: `cpu`).

#### Example

```
python evaluation.py --model clip --dataset tiny-c --corruption jpeg_compression --device cuda:0
```