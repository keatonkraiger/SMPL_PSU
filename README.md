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

Assuming you have the SOMA-derived models from the PSU dataset (the Subject_wise folder), you can create visualizations for each take with the `scripts/gen_viz.py` file. `gen_viz.py` has a global configuration `cfg` to set global parameters such as directories, sample rate, etc. Some important parameters include:

- `ds_rate`: The downsample rate for the frames. For example, if set to 5, every 5th frame will be processed.
- `n_jobs`: The number of parallel jobs to run for processing frames. This can speed up the processing significantly, especially on multi-core machines.
- `render_images`: If set to `True`, it will render images for each frame and generate a video from these images.
- `cleanup_imgs`: If set to `True`, it will remove the individual frame images after creating the video.
- `device`: The device to use for processing rendering, e.g., 'GPU' or 'CPU'.
- `camera_view`: The camera perspective to use for rendering, e.g., 1 or 2.

These values can be changed in the `cfg` dictionary at the top of the `gen_viz.py` file and a few may be set via command line arguments. Please see the code for more details.

To generate the take renderings as MP4s and a single blender animation for every subject-take, run the following command:

```bash
python scripts/gen_viz.py --render_images --create_animation
```