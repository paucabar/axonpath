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
pip install albumentations==1.3.1 monai==1.3.2 tensorboard==2.14.0 edt==2.4.1 bioimageio-core==0.9.0 bioimageio-spec==0.5.4.3
```

Install additional utilities:

```
conda install -c conda-forge pandas=1.5.3 tqdm colorcet matplotlib ipykernel -y
```
