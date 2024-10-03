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
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest ./demo/image
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
files based on the analysis you want to perform.  Below we show the steps to perform `3D Human Pose` analysis.

### Prerequisites

- Download mmdet checkpoint file

```shell
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth -P ./configs/mmdet/
```

- Download 2d checkpoint file

```shell
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth -P ./configs/2d/
```

- Download 3d checkpoint file

```shell
wget https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth -P ./configs/3d/
```

### Run Analysis

- Copy video to analyse into the `input` directory
- Run `python video-analysis --input input/<video-filename>`

Wait some time for job to finish and output will be placed within the `output` directory with the same filename as the input.

