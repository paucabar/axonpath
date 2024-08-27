# AimSeg-Monai_Panoptic

## Environment setup

Follow the steps described in the [Installation Guide]([url](https://docs.monai.io/en/stable/installation.html#installation-guide)) to enable GPU:

1. Install the latest NVIDIA driver (if not istalled).
2. Check [PyTorch Official Guide]([url](https://pytorch.org/get-started/locally/)) for the recommended CUDA versions.
3. Continue to follow the guide and install PyTorch.
4. Install MONAI using one the ways described below.

My installation for Windows (currently, PyTorch on Windows only supports Python 3.8-3.11)

```
conda create -n "monai" python=3.8
conda activate monai
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install monai
pip install tensorboard
```

Install dependencies:

```
conda install -c conda-forge albumentations
conda install -c conda-forge tqdm
conda install -c anaconda colorcet
conda install ipykernel
conda install conda-forge::matplotlib
conda install line_profiler
conda install -c conda-forge bioimageio.core=0.5.11
conda install -c conda-forge bioimageio.spec=0.4.9
```

For an improved performance during training install [MLAEDT-3D](https://github.com/seung-lab/euclidean-distance-transform-3d):

```
pip install edt
```
