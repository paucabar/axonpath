# AimSegDL

## Environment Setup

You have two options for setting up the environment:

### Option 1: Using the Provided Conda Environment (Recommended)

This ensures full compatibility with QuPath (tested with version 0.6.x and DJL 0.33.0):

```
conda env create -f envs/environment.yml
conda activate aimsegdl
```

### Option 2: Manual Installation (For Custom Setups)

1. Create base environment with PyTorch 2.5.1 installation — Recommended for QuPath integration:

```
conda create -n qupath-pytorch-251 python=3.9 pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda activate aimsegdl
```

You can also customize the CUDA version at: https://pytorch.org/get-started/locally/

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
