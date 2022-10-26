# Grasp'D: Differentiable Contact-rich Grasp Synthesis for Multi-fingered Hands

## Prerequisites

* Python 3.6 or higher
* PyTorch 1.4.0 or higher
* Pixar USD lib (optional, for advanced visualization)

Pre-built USD Python libraries can be downloaded from https://developer.nvidia.com/usd, once they are downloaded you should follow the corresponding instructions to add them to your `PYTHONPATH` environment variable. Besides using the provided basic visualizer implemented using pyvista, Grasp'D can generate USD files for rendering, e.g., in [NVIDIA Omniverse™](https://developer.nvidia.com/nvidia-omniverse-platform) or [usdview](https://graphics.pixar.com/usd/docs/USD-Toolset.html#USDToolset-usdview).

## Environment setup
Using anaconda set up a python environment with `conda env create -vv -f grasping/docker/environment.yml` or use `grasping/docker/Dockerfile` to build a docker image.

## Download dataset
You can download discretized SDFs for the YCB objects [here](https://drive.google.com/file/d/1aiYBeLI1HCYXVOfLwfochDDnwW1k_T9x/view?usp=sharing).
Unzip and copy these to `grasping/data/ycb`.
The folder structure should look like:
```
grasping/data/ycb
    |-- 002_master_chef_can
    |-- 003_cracker_box
    |-- ...
```

## Getting started
To collect grasps for the YCB banana, try running `python grasping/scripts/collect_grasps.py collector_config=011a_banana`.
See `grasping/conf/collect_grasps` for configuration options.
