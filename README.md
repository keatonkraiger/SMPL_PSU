# SMPL_PSU

This repository contains various scripts for working with the SMPL model and the PSU dataset.

## Installation

You should first install PyTorch and torchvision. You can do this by following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/). Next, you will need to install Blender's Python API, this can be done by installing the `bpy` package via pip.

To install the remaining packages, run:

```bash
pip install -e .
```

Once all the necessary packages are installed, you should download the [SMPL](https://smpl.is.tue.mpg.de/), [SMPLH](https://smpl.is.tue.mpg.de/), and [SMPLX](https://smpl-x.is.tue.mpg.de/) models from their respective websites. Place the downloaded files in the `body_models` directory. The directory structure should look like this:

```
.
├── body_models
│   ├── smpl
│   │   ├── SMPL_NEUTRAL.pkl
│   │   ├── SMPL_FEMALE.pkl
│   │   ├── SMPL_MALE.pkl
│   ├── smplh
│   │   ├── SMPLH_FEMALE.pkl
│   │   ├── SMPLH_MALE.pkl
│   │   ├── neutral
│   │   │   ├── model.npz
│   ├── smplx
│   │   ├── SMPLX_FEMALE.pkl
│   │   ├── SMPLX_MALE.pkl
│   │   ├── SMPLX_NEUTRAL.pkl
```

## Generating visualizations of SOMA data

Assuming you have the SOMA-derived models from the PSU dataset, you can create visualizations for each take with the `scripts/gen_viz.py` file. Currently, the script has a hardcoded cfg dict that specifies paths, params, etc. You can modify this to suit your needs. 
