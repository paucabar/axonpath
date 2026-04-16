# AxonPath

AxonPath is a deep learning framework for axon and myelin segmentation and morphometric analysis across microscopy modalities.

To learn more about the method, see the [paper – placeholder link] (preprint coming soon).
For hands-on usage, refer to the [tutorial notebook](notebooks/AxonPath_Tutorial_Pipeline.ipynb).

## Installation

Clone the repository:

```
git clone https://github.com/paucabar/aimseg-dl
cd aimseg-dl
```

Create the conda environment:

```
conda env create -f envs/environment.yml
conda activate axonpath
```

Requires Python 3.11 and a CUDA-capable GPU for training. Tested with PyTorch 2.7.1 and CUDA 12.8. You may adjust the `pytorch-cuda` version in `envs/environment.yml` to match your drivers — see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for compatible combinations. Data preparation and inference can run on CPU.

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

If your data is annotated in QuPath, you can export it directly in this format using the AxonPath training export [script](https://github.com/paucabar/qupath-daily/blob/main/export_annotations/axonwrap_training/export_labels_for_axonwrap.groovy).

Then run:

```
python scripts/prepare_data.py --input datasets --output prepared_data --create_test_split
```

This step will:

- Tile the input images
- Compute distance transforms
- Create train/validation/test splits under `prepared_data/`

### 2. Train a model

Train AxonPath on your prepared tiles:

```
python scripts/train_model.py --train_dir prepared_data/train_tiles --val_dir prepared_data/val_tiles --num_epochs 200 --batch_size 8 --min_diameter 30.0 --model_name my_axonpath_model
```

### 3. Evaluate the trained model

Run model evaluation on the test set:

```
python scripts/evaluate_model.py --test_dir prepared_data/test_tiles --model_path best_weights_model.pth --min_diameter 30.0 --output_csv Evaluation_Results --display_figure
```

See [`scripts/`](scripts/) for other available command-line examples.

## Example Usage (Full Pipeline Notebook)

- Follow the [AxonPath_Tutorial_Pipeline](notebooks/AxonPath_Tutorial_Pipeline.ipynb) for a complete interactive demonstration:
  - Prepare data
  - Train model
  - Evaluate segmentation results
  - Run inference
  - Measure axon/myelin morphometrics
  - Export a BioImage.IO-compliant model


## QuPath Integration

AxonPath can be used directly in QuPath via the
[AxonPath QuPath Extension](https://github.com/paucabar/qupath-extension-aimseg) (coming soon).

## Training Datasets

Raw imaging datasets and annotations used for developing AxonPath are available at:
[Dataset repository placeholder]

## Citation

If you use AxonPath in your work, please cite:

> **Carrillo-Barberà, P., Goldsborough, T., O'Callaghan, A., Poveda-Sabuco, A., Sgattoni, C.,**
> **Gomez-Sanchez, J.A., Rondelli, A., Williams, A., Bankhead, P.**
> *AxonWrap: multi-head segmentation for myelin analysis across microscopy modalities.*
> [Preprint placeholder link]
