# Image-SuperResolution-SREZ


Image super-resolution through deep learning. This project uses deep learning to upscale 16x16 images by a 4x factor.


## Dependencies:

- [Python v3.6.+ (64bits)](https://www.python.org/downloads/)
- NumPy v1.14.+
- Pillow v5.+
- OpenCV v3.14.+ (for cv2)
- MatPlotLib v2.1.+
- [TensorFlow v1.+](https://www.tensorflow.org/install/install_windows)

## Installations

Note: For install libraries use CMD terminal.

1. Download Python 3.6.+ (last version **of 64bits**), and install it. [Guide Video](https://www.youtube.com/watch?v=gSjL3K8C8Ao)
2. Installing numpy library (if not included)
```
py -m pip install numpy
```
3. Installing OpenCV library
```
py -m pip install opencv-python
```
4. Installing Pillow library
```
py -m pip install Pillow
```
5. Installing MatPlotLib
```
py -m pip install matplotlib
```
6. Installing TensorFlow (CPU or GPU) library
```
#*CPU version*
py -m pip install --upgrade tensorflow
```
or 
```
#*GPU version*
py -m pip install --upgrade tensorflow-gpu
```
For GPU Nvidia, must install [CUDA v9.0](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) and [cuDNN v9.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-windows10-x64-v7) (for cuDNN follow [this steps](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows))



## Project

Create a folder name `srez` and copy the following py files and folders:
- `checkpoint\` (learning process)
- `dataset\` (200K celebrity faces)
- `train\` (output of learning process, _empty_)
- `srez_demo.py` (Create Video animation of outputs)
- `srez_input.py` (dependicy of main.py)
- `srez_main.py` (main code)
- `srez_model.py` (dependicy of main.py)
- `srez_train.py` (For train the model with celeb faces)


## Using Networks

Usage is as follows:

1. Download 'img_align_celeba.zip' dataset 1.34GB from [celebA](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0). Unziped on dataset folder. This step may take a while as it will extract 200K images.

2. Training with default settings: `py srez_main.py --run train`. The script will periodically output an example batch in PNG format onto the srez/train folder, and checkpoint data will be stored in the srez/checkpoint folder.

3. After the network has trained you can also produce an animation showing the evolution of the output by running: `py srez_main.py --run demo`

_Note: In srez_main.py file, line 64, specify the training time (default: 1200 minutes). or any other values you want to change_

Reproduce: [David GPU](https://github.com/david-gpu/srez)
