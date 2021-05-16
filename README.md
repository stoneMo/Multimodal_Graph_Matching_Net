# 16824 Final Project: Multimodal Graph Matching Net

## Overview

Our implementation heavily relies on [InstanceRefer](https://github.com/CurryYuan/InstanceRefer). We follow their data preprocessing, training/testing setups. The environment setup and main procedure of data generation in this README is also adapt from that repository.

Our efforts include:
  - Implement our proposed MGMN model
  - Preprocess the language data into a tree structure using semantic parser.
  - Re-implemented/integrate [TGNN](https://github.com/hanhung/TGNN) in our repo under same setups. 
  - Re-organize the file structure. You can now deploy different model by specifying `model_name`.

## Getting Started

## Setup
The code is tested on Ubuntu 16.04 LTS & 18.04 LTS with PyTorch 1.3.0 CUDA 10.1 installed. 

```shell
conda install pytorch==1.3.0 cudatoolkit=10.1 -c pytorch
```

Install the necessary packages listed out in `requirements.txt`:
```shell
pip install -r requirements.txt
```
After all packages are properly installed, please run the following commands to compile the [torchsaprse](https://github.com/mit-han-lab/torchsparse):
```shell
cd lib/torchsparse/
python setup.py install
```
__Before moving on to the next step, please don't forget to set the project root path to the `CONF.PATH.BASE` in `lib/config.py`.__


### Data preparation
1. Download the ScanRefer dataset and unzip it under `data/`. 
2. Downloadand the preprocessed [GLoVE embeddings (~990MB)](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.
3. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](data/scannet/README.md) for downloading the ScanNet dataset). After this step, there should be folders containing the ScanNet scene data under the `data/scannet/scans/` with names like `scene0000_00`
4. Used official and pre-trained [PointGroup](https://github.com/Jia-Research-Lab/PointGroup) generate panoptic segmentation in `PointGroupInst/`. We provide pre-processed data in [Baidu Netdisk [password: 0nxc]](https://pan.baidu.com/s/1j9XCxPhaPECk4OczhjDxAA).
5. Pre-processed instance labels, and new data should be generated in  `data/scannet/pointgroup_data/`
```shell
cd data/scannet/
python prepare_data.py --split train --pointgroupinst_path [YOUR_PATH]
python prepare_data.py --split val   --pointgroupinst_path [YOUR_PATH]
python prepare_data.py --split test  --pointgroupinst_path [YOUR_PATH]
```
Finally, the dataset folder should be organized as follows.
```angular2
InstanceRefer
├── data
│   ├── scannet
│   │  ├── meta_data
│   │  ├── pointgroup_data
│   │  │  ├── scene0000_00_aligned_bbox.npy
│   │  │  ├── scene0000_00_aligned_vert.npy
│   │  ├──├──  ... ...

```
6. Please contact Shentong Mo (shentonm@andrew.cmu.edu) for our preprocessed parsing tree data. The npy file should be placed under `data/parsing_data`

### Training
Train the InstanceRefer/TGNN/MGMN model. You can change hyper-parameters in corresponding yaml file in `config/`:
```shell
python scripts/train.py --model_name [IR/TGNN/MGMN] --log_dir [your logger dir] --gpu [your gpu id]
```

## Acknowledgement
This project is not possible without multiple great opensourced codebases. 
* [ScanRefer](https://github.com/daveredrum/ScanRefer)
* [PointGroup](https://github.com/Jia-Research-Lab/PointGroup)
* [torchsaprse](https://github.com/mit-han-lab/torchsparse)
* [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
* [InstanceRefer](https://github.com/CurryYuan/InstanceRefer)
* [TGNN](https://github.com/hanhung/TGNN)

For any other problem, please kindly contact Jianchun Chen (jianchuc@andrew.cmu.edu) and Shentong Mo (shentonm@andrew.cmu.edu)