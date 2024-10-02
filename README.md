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
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest ./demo
python demo.py
```

This should output `PoseDataSample` within the logs

## Useful commands

To activate this environment, use                                                                                                                                                                           
                                                                                                                                                                                                             
     `conda activate openmmlab`                                                                                                                                                                              
                                                                                                                                                                                                             
To deactivate an active environment, use                                                                                                                                                                    
                                                                                                                                                                                                             
     `conda deactivate`