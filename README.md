# Deep Online Fused Video Stabilization

[[Paper]](https://openaccess.thecvf.com/content/WACV2022/papers/Shi_Deep_Online_Fused_Video_Stabilization_WACV_2022_paper.pdf)[[Supplementary]](https://zhmeishi.github.io/dvs/paper/dvs_supp.pdf)  [[Project Page]](https://zhmeishi.github.io/dvs/) [[Dataset]](https://storage.googleapis.com/dataset_release/all.zip) [[Our Result]](https://storage.googleapis.com/dataset_release/inference_result_release.zip) [[More Results]](https://zhmeishi.github.io/dvs/supp/results.html) 

This repository contains the Pytorch implementation of our method in the paper "Deep Online Fused Video Stabilization".

## Environment Setting
- Python version >= 3.10
- PyTorch with CUDA >= 2.2 (guide is [here](https://pytorch.org/get-started/locally/))

Install the remaining Python dependencies with the consolidated requirements file:
```
pip install -r requirements.txt
```

## Data Preparation
Download sample video [here](https://drive.google.com/file/d/1PpF3-6BbQKy9fldjIfwa5AlbtQflx3sG/view?usp=sharing).
Uncompress the *video* folder under the *dvs* folder.
```
python load_frame_sensor_data.py 
```
Demo of curve visualization:
The **gyro/OIS curve visualization** can be found at *dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820_real.jpg*.


## FlowNet2 Preparation
Note, we provide optical flow result of one test video in our Data Preparation. If you would like to generate them for all test videos, please follow [FlowNet2 official website](https://github.com/NVIDIA/flownet2-pytorch) and guide below. Otherwise, you can skip this section. 

Note, FlowNet2 installation is tricky. Ensure your CUDA toolkit matches the PyTorch wheels installed above (we target Python 3.10+ and PyTorch 2.2+). More details are [here](https://github.com/NVIDIA/flownet2-pytorch/issues/156) or contact us for any questions.

Download the FlowNet2 checkpoint into *dvs/flownet2* (the snippet below uses `torch.hub.download_url_to_file`, but any equivalent download method works):
```
python - <<'PY'
from pathlib import Path
from torch.hub import download_url_to_file

checkpoint_dir = Path("dvs/flownet2")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
download_url_to_file(
    "https://github.com/NVIDIA/flownet2-pytorch/releases/download/v1.0/FlowNet2_checkpoint.pth.tar",
    checkpoint_dir / "FlowNet2_checkpoint.pth.tar",
)
PY
```
```
python warp/read_write.py # video2frames
cd flownet2
bash install.sh # install package
bash run.sh # generate optical flow file for dataset
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

Follow FlowNet2 Preparation Section.
```
python warp/read_write.py --dir_path ./dataset_release # video2frames
cd flownet2
bash run_release.sh # generate optical flow file for dataset
``` 

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
