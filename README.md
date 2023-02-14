## CAIA
create time: 2023.02.12

#### Introduction
This repository contains the main source code for our cascade architecture model for Intracranial Aneurysm Detection in Computed Tomography Angiography Images, without model weights due to that it derived from a commercial software. The model is explained in the paper [xxx]().


Prerequisites
* Ubuntu: 16.04 lts
* Python 3.6.5
* Pytorch 1.8.2+cu102
* NVIDIA GPU + CUDA_10.2 CuDNN_7.5

This repository has been tested on NVIDIA RTX2080Ti. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

#### Installation

Other packages are as follows:
* yacs
* nibabel
* scipy
* joblib
* opencv-python
* SimpleITK
* scikit-image
* numpy

Install dependencies:
```shell script
pip install -r requirements.txt
```

#### Usage
You can use main.py as the entrance to this project.

The following is one example:
```shell script
python main.py --gpu 0 1 2 3  --test --config tasks/configs/aneurysm_seg.resunet.yaml
```
The main parameters are as following:

* --test: used to test(val) the model.
* --config: the path to the configuration file(*.yaml).
* --gpu(default 0): decide to which gpu to select. Format: one or multiple integers(separated by space keys), such as gpu 0 1 2 3


You can put the weight of the model to `{path_to_model}`

configuration file: `tasks/configs/aneurysm_seg.resunet.yaml`
* `TEST.DATA.NII_FOLDER`: directory of input files
* `TEST.DATA.TEST_FILE`: list of file names
* `TEST.SAVE_DIR`: the directory to save results
If you want to use your custom data, you need modify the yaml file to set the path and file names of the test data.
