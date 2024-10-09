# analyse-it

## Getting started

We are using the mmpose library for pose detection.  Steps to install below have been sourced from the following [installation guide](https://mmpose.readthedocs.io/en/latest/installation.html).

### Installation Steps

- Step 0. Download and install Miniconda from the [official website](https://docs.anaconda.com/miniconda/)

- Step 1. Create a conda environment and activate it.

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

- Step 2. Install PyTorch following official instructions, e.g.

```shell
conda install pytorch torchvision -c pytorch
```

- Step 3. Install MMEngine, MMCV and MMDET using mim

```shell
pip install -U openmim
mim install "mmengine==0.9.0"
mim install "mmcv==2.1.0"
mim install "mmdet==3.3.0"
```

- Step 4. Install MPose

```shell
mim install "mmpose==1.3.1"
```

- Step 5. Validate installation

```shell
python image-demo.py
```

This should output an annotated image, please check the output logs for its location.

## Useful commands

To activate this environment, use                                                                                                                                                                           
                                                                                                                                                                                                             
     `conda activate openmmlab`                                                                                                                                                                              
                                                                                                                                                                                                             
To deactivate an active environment, use                                                                                                                                                                    
                                                                                                                                                                                                             
     `conda deactivate`

## Analysing Video

Analysing video with [mmpose](https://mmpose.readthedocs.io/en/latest/) requires you to download different configuration
files based on the analysis you want to perform.  Below we show the steps to perform `2D Human Pose` analysis.

### Run Analysis

- Copy video to analyse into the `input` directory
- Run `python video-analysis.py --input input/<video-filename>`

Wait some time for job to finish and output will be placed within the `output`. Check logs for location.
Data that was used to generate analysis will also be output to `output/video/predictions`

