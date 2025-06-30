# AimSegDL

## Environment setup

Create a clean Python 3.8 environment

```
conda create -n aimsegdl python=3.8
conda activate aimsegdl
```

Install PyTorch with CUDA appropriate for your GPU: https://pytorch.org/get-started/locally/


Install core libraries:

```
pip install monai tensorboard albumentations==1.3.1 edt
```

Install additional utilities:

```
conda install -c conda-forge tqdm colorcet matplotlib ipykernel
```
