# AimSegDL

## Environment setup

You have two options to set up the environment:

### Option 1: Using the provided Conda environment (recommended)

This ensures full compatibility with GPU acceleration (CUDA, PyTorch, etc).

```
conda env create -f envs/environment-gpu.yml
conda activate aimsegdl
```

### Option 2: Manual installation (if you prefer pip or want to customize)

You can install dependencies manually using pip or conda, but CUDA/PyTorch compatibility is your responsibility.

```
conda create -n aimsegdl python=3.9
conda activate aimsegdl
```

Install PyTorch with CUDA appropriate for your GPU: https://pytorch.org/get-started/locally/


Install core libraries:

```
pip install albumentations==1.3.1 monai==1.3.2 tensorboard==2.14.0 edt==2.4.1 bioimageio-core==0.9.0 bioimageio-spec==0.5.4.3
```

Install additional utilities:

```
conda install -c conda-forge numpy=1.26 pandas=1.5.3 tqdm colorcet matplotlib ipykernel -y
```