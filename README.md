
# Joint-ID: Transformer-Based Joint Image Enhancement and Depth Estimation for Underwater Environments

**IEEE Sensors Journal 2023**

This repository represents the official implementation of the paper titled "Transformer-Based Joint Image Enhancement and Depth Estimation for Underwater Environments".

[![ProjectPage](fig/badges/badge-website.svg)](https://sites.google.com/view/joint-id/home)
[![Paper](https://img.shields.io/badge/📄%20Paper-PDF-yellow)](https://ieeexplore.ieee.org/abstract/document/10351035)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=1GOoHCSpNXE&t=3s) 
[![Docker](https://badges.aleen42.com/src/docker.svg)](https://hub.docker.com/r/ygm7422/official_joint_id)
[![License](https://img.shields.io/badge/License-GPL--3.0-929292)](https://www.gnu.org/licenses/gpl-3.0.html)

[Geonmo Yang](https://scholar.google.com/citations?user=kiBTkqMAAAAJ&hl=en&oi=sra),
[Gilhwan Kang](https://scholar.google.com/citations?user=F6dY8DoAAAAJ&hl=ko),
[Juhhui Lee](https://scholar.google.com/citations?user=4-5Fi9kAAAAJ&hl=en),
[Younggun Cho](https://scholar.google.com/citations?user=W5MOKWIAAAAJ&hl=ko)

We propose a novel approach for enhancing underwater images that leverages the benefits of joint learning for simultaneous image enhancement and depth estimation. We introduce **Joint-ID**, a transformer-based neural network that can obtain high-perceptual image quality and depth information from raw underwater images. Our approach formulates a multi-modal objective function that addresses invalid depth, lack of sharpness, and image degradation based on color and local texture.

![teaser](fig/joint_id.png)

## 🛠️ Prerequisites
1. Run the demo locally (requires a GPU and an `nvidia-docker2`, see [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

2. Optionally, we provide instructions to use docker in multiple ways. (But, recommended using `docker compose`, see [Installation Guide](https://docs.docker.com/compose/install/linux/)).

3. The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. But, we don't provide the instructions to install both PyTorch and TorchVision dependencies. Please use `nvidia-docker2` 😁. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

4. This code was tested on:
  - Ubuntu 22.04 LTS, Python 3.10.12, CUDA 11.7, GeForce RTX 3090 (pip)
  - Ubuntu 22.04 LTS, Python 3.8.6, CUDA 12.0, RTX A6000 (pip)
  - Ubuntu 20.04 LTS, Python 3.10.12, CUDA 12.1, GeForce RTX 3080ti (pip)

## 🚀 Contents Table
### 🛠️ Setup
1. [**📦 Prepare Repository & Checkpoints**](https://github.com/sparolab/Joint_ID?tab=readme-ov-file#-prepare-repository--checkpoints)
2. [**⬇  Prepare Dataset**](https://github.com/sparolab/Joint_ID?tab=readme-ov-file#-prepare-dataset)
3. [**🐋 Prepare Docker Image and Run the Docker Container**](https://github.com/sparolab/Joint_ID?tab=readme-ov-file#-prepare-docker-image-and-run-the-docker-container)

### 🚀 Traning or Testing for Joint-ID
1. [**🚀 Training for Joint-ID on Joint-ID Dataset**](https://github.com/sparolab/Joint_ID?tab=readme-ov-file#-training-for-joint-id-on-joint-id-dataset)
2. [**🚀 Testing for Joint-ID on Joint-ID Dataset**](https://github.com/sparolab/Joint_ID?tab=readme-ov-file#-testing-for-joint-id-on-joint-id-dataset)
3. [**🚀 Testing for Joint-ID on Standard or Custom Dataset**](https://github.com/sparolab/Joint_ID?tab=readme-ov-file#-testing-for-joint-id-on-standard-or-custom-dataset)

### ✏️ ETC
4. [**⚙️ Inference settings**](https://github.com/sparolab/Joint_ID?tab=readme-ov-file#%EF%B8%8F-inference-settings) 

5. [**🎓 Citation**](https://github.com/sparolab/Joint_ID?tab=readme-ov-file#-citation)

<br/>

## 🛠️ Setup
### 📦 Prepare Repository & Checkpoints
1. Clone the repository (requires git):

    ```bash
    git clone https://github.com/sparolab/Joint_ID.git
    cd Joint_ID
    ```

2. Let's call the path where Joint-ID's repository is located `${Joint-ID_root}`.

3. Download a checkpoint [**joint_id_ckpt.pth**](https://www.dropbox.com/scl/fo/rn49h1r54uqsdsjs896jf/h?rlkey=u0yypv3y7y5lm20a81vqcjyxm&dl=0) of our model on path `${Joint-ID_root}/Joint_ID`.

<br/>

### ⬇ Prepare Dataset
![dataset](fig/dataset.png)
1. Download the [**Joint_ID_Dataset.zip**](https://www.dropbox.com/scl/fo/olr8awsue6uion5fng25j/h?rlkey=jy6pbnbop6ppc0703it7lmite&dl=0)

2. Next, unzip the file named `Joint_ID_Dataset.zip` with the downloaded path as `${dataset_root_path}`.
    ```bash

    sudo unzip ${dataset_root_path}/Joint_ID_Dataset.zip   # ${dataset_root_path} requires at least 2.3 Gb of space.
    # ${dataset_root_path} is the absolute path, not relative path.
    ```

3. After downloading, you should see the following file structure in the `Joint_ID_Dataset` folder
    ```
    📦 Joint_ID_Dataset
    ┣ 📂 train
    ┃ ┣ 📂 LR                  # GT for traning dataset
    ┃ ┃ ┣ 📂 01_Warehouse  
    ┃ ┃ ┃ ┣ 📂 color           # enhanced Image
    ┃ ┃ ┃ ┃ ┣ 📜 in_00_160126_155728_c.png
    ┃ ┃ ┃ ┃       ...
    ┃ ┃ ┃ ┃
    ┃ ┃ ┃ ┗ 📂 depth_filled    # depth Image
    ┃ ┃ ┃   ┣ 📜 in_00_160126_155728_depth_filled.png
    ┃ ┃ ┃         ...
    ┃ ┃ ...
    ┃ ┗ 📂 synthetic           # synthetic distorted dataset
    ┃   ┣ 📜 LR@01_Warehouse@color...7512.jpg
    ┃   ┣      ...
    ┃  
    ┗ 📂 test         # 'test'folder has same structure as 'train'folder
          ...
    ```
4. After downloading, you should see the following file structure in the `Joint_ID_Dataset` folder

5. If you want to know the dataset, then see the [project page](https://sites.google.com/view/joint-id/home) for additional dataset details.

<br/>

### 🐋 Prepare Docker Image and Run the Docker Container
To run a docker container, we need to create a docker image. There are two ways to create a docker image and run the docker container.

1. Use `docker pull` or:

    ```bash
    # download the docker image
    docker pull ygm7422/official_joint_id:latest    
    
    # run the docker container
    nvidia-docker run \
    --privileged \
    --rm \
    --gpus all -it \
    --name joint-id \
    --ipc=host \
    --shm-size=256M \
    --net host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=unix$DISPLAY \
    -v /root/.Xauthority:/root/.Xauthority \
    --env="QT_X11_NO_MITSHM=1" \
    -v ${dataset_root_path}/Joint_ID_Dataset:/root/workspace/dataset_root \
    -v ${Joint-ID_root}/Joint_ID:/root/workspace \
    ygm7422/official_joint_id:latest 
    ```

2. Use `docker compose` (this is used to build docker iamges and run container simultaneously):

    ```bash
    cd ${Joint-ID_root}/Joint_ID

    # build docker image and run container simultaneously
    bash run_docker.sh up gpu ${dataset_root_path}/Joint_ID_Dataset
    
    # Inside the container
    docker exec -it Joint_ID bash
    ```

Regardless of whether you use method 1 or 2, you should have a docker container named `Joint_ID` running.

<br/>

## 🚀 Traning or Testing for Joint-ID 

### 🚀 Training for Joint-ID on Joint-ID Dataset
1. First, move to the `/root/workspace` folder inside the docker container. Then, run the following command to start the training.
    ```bash
    # move to workspace
    cd /root/workspace

    # start to train on Joint-ID Dataset
    python run.py local_configs/arg_joint_train.txt
    ```
2. The model's checkpoints and log files are saved in the `/root/workspace/save` folder.
3. If you want to change the default variable setting for training, see [**Inference settings**](https://github.com/sparolab/Joint_ID?tab=readme-ov-file#%EF%B8%8F-inference-settings) below.

<br/>

### 🚀 Testing for Joint-ID on Joint-ID Dataset
1. First, move to the `/root/workspace` folder inside the docker container. Then, run the following command to start the testing.
    ```bash
    # move to workspace
    cd /root/workspace

    # start to test on Joint-ID Dataset
    python run.py local_configs/arg_joint_test.txt
    ```
2. The test images and results are saved in the `result_joint.diml.joint_id` folder.

3. If you want to change the default variable setting for testing, see [**Inference settings**](https://github.com/sparolab/Joint_ID?tab=readme-ov-file#%EF%B8%8F-inference-settings) below.

<br/>

### 🚀 Testing for Joint-ID on Standard or Custom Dataset
1. Set the dataset related variables in the `local_configs/cfg/joint.diml.joint_id.py` file. Below, enter the input image path in the `sample_test_data_path` variable.
    ```python
    ...

    # If you want to adjust the image size, adjust the `image_size` below.
    image_size = dict(input_height=288,
                      input_width=512)
    ...

    # Dataset
    dataset = dict(
               train_data_path='dataset_root/train/synthetic',
               ...
               # sample_test_data_path='${your standard or custom dataset path}',
               sample_test_data_path='demo',
               video_txt_file=''
               )
    ...
    ```
    
2. First, move to the `/root/workspace` folder inside the docker container. Then, run the following command to start the testing.
    ```bash
    # move to workspace
    cd /root/workspace
    
    # start to test on standard datasets
    python run.py local_configs/arg_joint_samples_test.txt
    ```

3. The test images and results are saved in the `sample_eval_result_joint.diml.joint_id` folder.

<br/>

## ⚙️ Inference settings
We set the hyperparameters in 'local_configs/cfg/joint.diml.joint_id.py'. 

`depth_range`: Range of depth we want to estimate

`image_size`: the size of the input image data. If you set this variable, make sure to set `auto_crop` to False in `train_dataloader_cfg`, or `eval_dataloader_cfg`, or `test_dataloader_cfg`, or `sample_test_cfg` below. If you do not want to set `image_size`, please set `auto_crop` to True. `auto_crop` will be input to the model at the original size of the input data.

`train_parm`: hyperparameters to set when training.

`test_parm`: hyperparameters to set when testing.


<br/>

## 🎓 Citation
Please cite our paper:
```bibtex
@article{yang2023joint,
  title={Joint-ID: Transformer-based Joint Image Enhancement and Depth Estimation for Underwater Environments},
  author={Yang, Geonmo and Kang, Gilhwan and Lee, Juhui and Cho, Younggun},
  journal={IEEE Sensors Journal},
  year={2023},
  publisher={IEEE}
}
```

<br/>

##  🎫 License
This work is licensed under the GPL License, Version 3.0 (as defined in the [LICENSE](LICENSE)).

[![License](https://img.shields.io/badge/License-GPL--3.0-929292)](https://www.gnu.org/licenses/gpl-3.0.html)
