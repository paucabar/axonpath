# AimSegDL

AimSegDL is a deep learning framework for axon and myelin segmentation and morphometric analysis across microscopy modalities.

To learn more about the method, see the [paper – placeholder link] (preprint coming soon).
For hands-on usage, refer to the [tutorial notebook](notebooks/AimSegDL_Tutorial_Pipeline.ipynb).

## Installation

Clone the repository:

```
git clone https://github.com/paucabar/aimseg-dl
cd aimseg-dl
```

You have two options for setting up the environment:

### Option 1: Using the Provided Conda Environment

This ensures full compatibility with QuPath (tested with version 0.6.x and DJL 0.33.0):

```
conda env create -f envs/environment.yml
conda activate aimsegdl
```

### Option 2: Manual Installation

1. Create base environment with PyTorch 2.5.1 installation — Recommended for QuPath integration:

```
conda create -n qupath-pytorch-251 python=3.9 pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda activate aimsegdl
```

You may adjust CUDA and PyTorch versions as listed on [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

```
conda create -n qupath-pytorch-251 python=3.9 pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

2. Install core libraries:

```
pip install albumentations==1.3.1 monai==1.3.2 tensorboard==2.14.0 edt==2.4.1 bioimageio-core==0.9.0 bioimageio-spec==0.5.4.3
```

3. Install additional utilities:

```
conda install -c conda-forge numpy=1.26 pandas=1.5.3 tqdm colorcet matplotlib ipykernel -y
```

## Quickstart (Command-Line Workflow)

### 1. Prepare your dataset

Before training, organize your data in the following structure (each image must have a corresponding mask and label with the same filename):

```
datasets/
  dataset1/
    images/
    masks/
    labels/
```

If your data is annotated in QuPath, you can export it directly in this format using the axonwrap_training export [script](https://github.com/paucabar/qupath-daily/blob/main/export_annotations/axonwrap_training/export_labels_for_axonwrap.groovy)

Then run:

```
python prepare_data.py --input datasets --output prepared_data --create_test_split
```

This step will:

- Tile the input images
- Compute distance transforms
- Create train/validation/test splits under `prepared_data/`

### 2. Train a model

Train AimSegDL on your prepared tiles:

```
python train_model.py --train_dir prepared_data/train_tiles --val_dir prepared_data/val_tiles --num_epochs 200 --batch_size 8 --min_diameter 30.0 --model_name my_aimsegdl
```

### 3. Evaluate the trained model

Run model evaluation on the test set:

```
python evaluate_model.py --test_dir prepared_data/test_tiles --model_path best_weights_model.pth --min_diameter 30.0 --output_csv Evaluation_Results --display_figure
```


See [`scripts/`](scripts/) for other available command-line examples.

## Example Usage (Full Pipeline Notebook)

- Follow the [AimSegDL_Tutorial_Pipeline](notebooks/AimSegDL_Tutorial_Pipeline.ipynb) for a complete interactive demonstration:
  - Prepare data
  - Train model
  - Evaluate segmentation results
  - Run inference
  - Measure axon/myelin morphometrics
  - Export a BioImage.IO-compliant model


## QuPath Integration

AimSegDL can be used directly in QuPath via the
[AimSegDL-QuPath Extension](https://github.com/paucabar/qupath-extension-aimseg) (coming soon)

## Training Datasets

Raw imaging datasets and annotations used for developing AimSegDL are available at:
[Dataset repository placeholder]

## Citation

If you use AimSegDL in your work, please cite:

> **Carrillo-Barberà, P., Goldsborough, T., O'Callaghan, A., Poveda-Sabuco, A., Sgattoni, C.,**
> **Gomez-Sanchez, J.A., Rondelli, A., Williams, A., Bankhead, P.**
> *AxonWrap: multi-head segmentation for myelin analysis across microscopy modalities.*
> [Preprint placeholder link]
