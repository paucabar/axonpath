# AxonPath

AxonPath is a deep learning framework for segmenting myelinated nerve fibres, and measuring their
morphometry, in electron microscopy (EM) and brightfield images.

This repository contains the Python package used to prepare training data, train and evaluate
AxonPath models, run inference, compute morphometric measurements and export models in
[BioImage.IO](https://bioimage.io) format. To run the pretrained models without code, use the
[AxonPath QuPath extension](#qupath-integration).

## How it works

AxonPath uses a U-Net with three output heads:

- a **semantic head** that classifies each pixel as background, myelin or inner cylinder
  (the axon plus the inner tongue of the myelin sheath);
- a **fibre distance head** and an **axon distance head** that predict skeleton-aware distance
  transforms (SDTs), which are split into individual fibres and axons by seeded watershed.

The result is a hierarchy of objects, **Fibre > InnerCylinder > Axon**, from which g-ratios and
other morphometric measurements are computed. Axons always come from the axon distance head and
inner cylinders from the semantic head. In brightfield images the inner tongue cannot be resolved,
so only fibres and axons are segmented (**Fibre > Axon**), and the axon includes the inner tongue.

## Pretrained models

Two sets of pretrained weights are included in the package and can be loaded by name:

| Name | Images | `min_diameter` (px) |
|---|---|---|
| `em` | Electron microscopy | 20 |
| `bf` | Brightfield | 15 |

They are also distributed as three BioImage.IO packages on Zenodo, for use in QuPath:
**[doi.org/10.5281/zenodo.22982902](https://doi.org/10.5281/zenodo.22982902)**.

| Package | Weights | Images | Pixel size |
|---|---|---|---|
| `em_cns-0.1.0` | `em` | EM, central nervous system | 0.008 µm |
| `em_pns-0.1.0` | `em` | EM, peripheral nervous system | 0.03 µm |
| `brightfield-0.1.0` | `bf` | Brightfield | 0.071 µm |

The models work best on images at their training pixel size. When using the weights from Python,
rescale your images to that pixel size first; `min_diameter` is given in pixels at that size. The
example images in [`axonpath/example_images/`](axonpath/example_images/) are the test images of the
three packages, already at the right pixel size.

## Installation

Clone the repository:

```
git clone https://github.com/paucabar/axonpath
cd axonpath
```

Create the conda environment. This also installs the `axonpath` package itself in editable mode:

```
conda env create -f envs/environment.yml
conda activate axonpath
```

Requires Python 3.11. A CUDA-capable GPU is strongly recommended for training; data preparation
and inference can run on CPU. Tested with PyTorch 2.7.1 and CUDA 12.8. You may adjust the
`pytorch-cuda` version in `envs/environment.yml` to match your drivers; see
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for compatible
combinations.

Optional extras, installed with pip into the active environment:

```
pip install -e ".[hpo]"   # hyperparameter optimisation (Optuna)
pip install -e ".[dev]"   # test suite (pytest)
```

## Quickstart (command line)

### 1. Prepare your dataset

Organise your data as follows. Each image must have a mask and a label image with the same file
name:

```
datasets/
  dataset1/
    images/    # the images
    masks/     # semantic masks: 0 background, 1 myelin, 2 inner tongue, 3 axon
    labels/    # fibre instance labels: one integer ID per fibre
  dataset2/
    ...
```

If your data are annotated in QuPath, the AxonPath extension exports them in exactly this format:
see [Export annotations → Training mode](https://github.com/paucabar/qupath-extension-axonpath/blob/main/docs/export.md#training-mode).
For training, values 2 and 3 together form the inner cylinder. Brightfield masks have no inner
tongue (values 0, 1 and 3 only).

Then run:

```
python scripts/prepare_data.py --input datasets --output prepared_data --create_test_split
```

This step:

- tiles the input images;
- computes the skeleton-aware distance transforms used as training targets;
- creates train, validation and (optionally) test splits under `prepared_data/`.

To check annotations for common problems before preparing them, run
`python scripts/check_annotations.py --input datasets`.

### 2. Train a model

```
python scripts/train_model.py --train_dir prepared_data/train_tiles --val_dir prepared_data/val_tiles --num_epochs 1000 --batch_size 8 --min_diameter 20 --model_name my_axonpath_model
```

To fine-tune from the pretrained weights instead of training from scratch, add
`--pretrained_weights em` (or `bf`, or the path to a `.pth` file). Run
`python scripts/train_model.py --help` for all options.

### 3. Evaluate the trained model

```
python scripts/evaluate_model.py --test_dir prepared_data/test_tiles --model_path best_weights_model.pth --min_diameter 20 --output_csv Evaluation_Results --display_figure
```

This reports F1, precision and recall for fibres, inner cylinders and axons, with objects matched
at IoU thresholds from 0.5 to 0.9.

### 4. Export for QuPath

```
python scripts/export_bioimageio.py --model_path best_weights_model.pth --test_img_path my_test_image.tif --model_name my_model --pixel_size 0.008 --min_diameter 20 --predict_inner_cylinder --output_dir my_model-0.1.0
```

Omit `--predict_inner_cylinder` for brightfield models. The output folder can be placed in the
QuPath extension's model directory. Add `--validate` to check the package against the BioImage.IO
specification.

Other scripts in [`scripts/`](scripts/):

| Script | Purpose |
|---|---|
| `prepare_and_train.py` | Data preparation and training in one command |
| `optimise_hparams.py` | Hyperparameter search with Optuna (requires the `hpo` extra) |
| `check_annotations.py` | Quality control of raw annotations |

## Tutorial notebook

The [AxonPath tutorial](notebooks/AxonPath_Tutorial_Pipeline.ipynb) runs the whole workflow from
Python:

- prepare data;
- train a model;
- evaluate it on a test set;
- run inference with the pretrained or your own weights;
- measure fibre, inner cylinder and axon morphometry (g-ratios, shape, spatial metrics);
- export a BioImage.IO model.

Two further notebooks help to check the training data:
[Inspect_Prepared_Tiles](notebooks/Inspect_Prepared_Tiles.ipynb) shows prepared tiles and their
targets, and [Inspect_Augmentations](notebooks/Inspect_Augmentations.ipynb) shows the training
augmentations.

## QuPath integration

The [AxonPath QuPath extension](https://github.com/paucabar/qupath-extension-axonpath) runs
AxonPath models in QuPath, lets you review and correct the results, and exports measurements and
training data. It is installed through QuPath's Extension Manager; see its
[documentation](https://github.com/paucabar/qupath-extension-axonpath#readme).

## Training datasets

Raw imaging datasets and annotations used for developing AxonPath are available at:
[Dataset repository placeholder]

## Running the tests

```
pip install -e ".[dev]"
python -m pytest
```

## Citation

A preprint describing AxonPath is in preparation. Citation details will be added here.

To cite the models, use their Zenodo record:
[doi.org/10.5281/zenodo.22982902](https://doi.org/10.5281/zenodo.22982902).

## License

The source code is released under the [Apache License 2.0](LICENSE).

The pretrained model weights (`axonpath/weights/`, also distributed as BioImage.IO packages on
[Zenodo](https://doi.org/10.5281/zenodo.22982902)) and the example images
(`axonpath/example_images/`) are released under the
[Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)
license.
