# TinySeg Model Optimizing Framework

## Introduction

TinySeg is a model optimizing framework for memory-efficient image segmentation. It optimizes the peak memory usage of the input model by spilling a long-living tensor into the local or remote storage and removing large interim tensors for concatenation.

## Installation

The prototype implementation requires the following packages:
- tensorflow-cpu
- absl-py
- numpy
- flatbuffers

```bash
python3 -m pip install -r requirements.txt
```

## Model Optimization

If you have a TensorFlow Lite model, you can optimize the model using the following command:

```bash
python ./optimizer/optimizer.py --model [path to model] --target [target memory usage]
```

Note that the optimizer will generate the optimzied model in the folder that contains the input model.

## Application

You can test the model using the provided Arduino application, which contains the implementation of the TinySeg runtime. You will need to download the modified TensorFlow Lite library at this [link](https://drive.google.com/file/d/1nq3fluy_hChEBcyKHOO_0PvAPDjdFyoh/view?usp=sharing). To install the library, unzip the archive file and place the folder in the Arduino libraries folder.

## Citation

```
@inproceedings{chae:lctes:2024,
  author = {Chae, Byungchul and Kim, Jiae and Heo, Seonyeong},
  title = {TinySeg: Model Optimizing Framework for Image Segmentation on Tiny Embedded Systems},
  year = {2024},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  booktitle = {Proceedings of the 25th ACM SIGPLAN/SIGBED Conference on Languages, Compilers, and Tools for Embedded Systems},
  series = {LCTES 2024}
}
```
