## CAIA
create time: 2023.02.12

#### Introduction
This repository contains the main source code for our cascade architecture model for Intracranial Aneurysm Detection in Computed Tomography Angiography Images, without model weights due to that it derived from a **commercial software**. The model is explained in the paper [xxx]().


Prerequisites
* Ubuntu: 18.04 lts
* Python 3.9.7
* Pytorch 1.8.2+cu102
* NVIDIA GPU + CUDA_10.2 CuDNN_7.5

This repository has been tested on NVIDIA TITANXP. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

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
# 1.resunet 3d train/eval
# train
python main.py --gpu 0 1 2 3  --train --config tasks/configs/aneurysm_seg.resunet_titanxp4.yaml

# eval
python main.py --gpu 0 1 2 3  --test --config tasks/configs/aneurysm_seg.resunet_titanxp4.yaml --check_point /path/to/model

# 2.densenet 3d train/eval
# train
python main.py --gpu 0 1 2 3  --train --config tasks/configs/aneurysm_cls.densenet_titanxp4.yaml

# eval
python main.py --gpu 0 1 2 3  --test --config tasks/configs/aneurysm_cls.densenet_titanxp4.yaml --check_point /path/to/model

```
The main parameters are as following:
* --train: used to train your model.
* --config: the path to the configuration file(*.yaml).
* --gpu(default 0): decide to which gpu to select. Format: one or multiple integers(separated by space keys), such as `--gpu 0 1 2 3`
* --test: used to test(val) the model.
* --check_point: checkpoints/model path, used to evaluate in test stage
  * You can put the weight of the model to `{/path/to/model}`

configuration file:
* task1: `tasks/configs/aneurysm_seg.resunet_titanxp4.yaml`
    * `TEST.DATA.NII_FOLDER`: directory of input files
    * `TEST.DATA.TEST_FILE`: list of file names
    * `TEST.SAVE_DIR`: the directory to save results
* task2: `tasks/configs/aneurysm_cls.densenet_titanxp4.yaml`
  * `TEST_IMAGE_DIR`: directory of input files
  * `TEST.DATA.TEST_LIST`: list of  suspicious aneurysms lesion patch file path and label(0,1)
If you want to use your custom data, you need modify the yaml file to set the path and file names of the test data.


#### demo dataset

We show two tasks's demo dataset in `./raws/demo_dataset`
- aneurysm_ds_seg: for task1 resunet 3d
- aneurysm_ds_cls: for task2 densenet 3d


