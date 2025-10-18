# Deep Online Fused Video Stabilization

[[Paper]](https://openaccess.thecvf.com/content/WACV2022/papers/Shi_Deep_Online_Fused_Video_Stabilization_WACV_2022_paper.pdf)[[Supplementary]](https://zhmeishi.github.io/dvs/paper/dvs_supp.pdf)  [[Project Page]](https://zhmeishi.github.io/dvs/) [[Dataset]](https://storage.googleapis.com/dataset_release/all.zip) [[Our Result]](https://storage.googleapis.com/dataset_release/inference_result_release.zip) [[More Results]](https://zhmeishi.github.io/dvs/supp/results.html) 

This repository contains the Pytorch implementation of our method in the paper "Deep Online Fused Video Stabilization".

## Environment Setting
Python version >= 3.6
Pytorch with CUDA >= 1.0.0 (guide is [here](https://pytorch.org/get-started/locally/))
Install other used packages:
```
cd dvs
pip install -r requirements.txt --ignore-installed
# Install/update torchvision to access the bundled RAFT model
pip install "torchvision>=0.13"
```

## Data Preparation
Download sample video [here](https://drive.google.com/file/d/1PpF3-6BbQKy9fldjIfwa5AlbtQflx3sG/view?usp=sharing).
Uncompress the *video* folder under the *dvs* folder.
```
python load_frame_sensor_data.py 
```
Demo of curve visualization:
The **gyro/OIS curve visualization** can be found at *dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820_real.jpg*.


## Optical Flow Backend
The project now relies on the RAFT implementation that ships with `torchvision` to generate optical flow. The wrapper in
`dvs/flow/raft_wrapper.py` downloads the pre-trained weights automatically the first time you invoke training or inference. No
manual compilation or `.flo` preprocessing is required any more.

Make sure your environment has a recent `torchvision` build (e.g., `pip install "torchvision>=0.13"`). When processing large
sequences you can still export frames up front for faster access:
```
python warp/read_write.py  # optional: converts videos to frame folders
```

## Running Inference 
```
python inference.py
python metrics.py
``` 
The loss and metric information will be printed in the terminal. The metric numbers can be slightly different due to difference on opencv/pytorch versions.  

The result is under *dvs/test/stabilzation*.   
In *s_114_outdoor_running_trail_daytime.jpg*, the blue curve is the output of our models, and the green curve is the input.   
*s_114_outdoor_running_trail_daytime_stab.mp4* is uncropped stabilized video.  
*s_114_outdoor_running_trail_daytime_stab_crop.mp4* is cropped stabilized video. Note, the cropped video is generated after running the metrics code.   

## Training
Download dataset for training and test [here](https://storage.googleapis.com/dataset_release/all.zip). 
Uncompress *all.zip* and move *dataset_release* folder under the *dvs* folder.

Run training code.
```
python train.py
```
The model is saved in *checkpoint/stabilzation_train*.

## Citation 
If you use this code or dataset for your research, please cite our paper.
```
@inproceedings{shi2022deep,
  title={Deep Online Fused Video Stabilization},
  author={Shi, Zhenmei and Shi, Fuhao and Lai, Wei-Sheng and Liang, Chia-Kai and Liang, Yingyu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1250--1258},
  year={2022}
}
```
