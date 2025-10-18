from torch.utils.data import Dataset
import os
import collections
from gyro import (
    LoadGyroData,
    LoadOISData,
    LoadFrameData,
    GetGyroAtTimeStamp,
    get_static,
    GetMetadata,
    GetProjections,
    train_GetGyroAtTimeStamp,
    QuaternionProduct,
    QuaternionReciprocal,
    FindOISAtTimeStamp,
    norm_quat
    )
import random
import numpy as np
import torchvision.transforms as transforms
import torch
import cv2
from scipy import ndimage, misc
from numpy import linalg as LA
from flow.raft_wrapper import RAFTFlowEstimator

def get_data_loader(cf, no_flo = False):
    size = cf["data"]["batch_size"]
    num_workers = cf["data"]["num_workers"]
    train_data, test_data = get_dataset(cf, no_flo)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=size,shuffle=True, pin_memory=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=size,shuffle=False, pin_memory=True, num_workers=num_workers)
    return trainloader,testloader

def get_dataset(cf, no_flo = False):
    resize_ratio = cf["data"]["resize_ratio"]
    train_transform, test_transform = _data_transforms()
    train_path = os.path.join(cf["data"]["data_dir"], "training")
    test_path = os.path.join(cf["data"]["data_dir"], "test")
    if not os.path.exists(train_path):
        train_path = cf["data"]["data_dir"]
    if not os.path.exists(test_path):
        test_path = cf["data"]["data_dir"]

    train_data = Dataset_Gyro(
        train_path, sample_freq = cf["data"]["sample_freq"]*1000000, number_real = cf["data"]["number_real"], 
        time_train = cf["data"]["time_train"]*1000000, transform = train_transform, resize_ratio = resize_ratio, no_flo = no_flo)
    test_data = Dataset_Gyro(
        test_path, sample_freq = cf["data"]["sample_freq"]*1000000, number_real = cf["data"]["number_real"], 
        time_train = cf["data"]["time_train"]*1000000, transform = test_transform, resize_ratio = resize_ratio, no_flo = no_flo)
    return train_data, test_data

def get_inference_data_loader(cf, data_path, no_flo = False):
    test_data = get_inference_dataset(cf, data_path, no_flo)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=False, pin_memory=True, num_workers=1)
    return testloader

def get_inference_dataset(cf, data_path, no_flo = False):
    resize_ratio = cf["data"]["resize_ratio"]
    _, test_transform = _data_transforms()
    test_data = Dataset_Gyro(
        data_path, sample_freq = cf["data"]["sample_freq"]*1000000, number_real = cf["data"]["number_real"], 
        time_train = cf["data"]["time_train"]*1000000, transform = test_transform, resize_ratio = resize_ratio,
        inference_only = True, no_flo = no_flo)
    return test_data

def _data_transforms():

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        ])
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        ])

    return train_transform, test_transform

class DVS_data():
    def __init__(self):
        self.gyro = None
        self.ois = None
        self.frame = None
        self.length = 0
        self.flo_shape = None
        self.frame_paths = None
        self.video_path = None
        self._frame_cache = None

class Dataset_Gyro(Dataset):
    def __init__(self, path, sample_freq = 33*1000000, number_real = 10, time_train = 2000*1000000, \
        transform = None, inference_only = False, no_flo = False, resize_ratio = 1):
        r"""
        Arguments:
            sample_freq: real quaternions [t-sample_freq*number_real, t+sample_freq*number_real] ns
            number_real: real gyro num in half time_interval
            time_train: time for a batch ns
        """
        self.sample_freq = sample_freq
        self.number_real = number_real
        self.no_flo = no_flo
        self.resize_ratio = resize_ratio
        self.static_options = get_static()
        self.inference_only = inference_only
        self.flow_wrapper = None

        self.ois_ratio = np.array([self.static_options["crop_window_width"] / self.static_options["width"], \
            self.static_options["crop_window_height"] / self.static_options["height"]]) * 0.01
        self.unit_size = 4

        if inference_only:
            self.length = 1
            self.data = [self.process_one_video(path)]
            self.number_train = self.data[0].length  
            return

        self.time_train = time_train
        self.number_train = time_train//self.sample_freq
        
        self.data_name = sorted(os.listdir(path))
        self.length = len(self.data_name)
        self.data = []
        for i in range(self.length):
            self.data.append(self.process_one_video(os.path.join(path,self.data_name[i])))

    def process_one_video(self, path):
        dvs_data = DVS_data()
        files = sorted(os.listdir(path))
        print(path)
        for f in files:
            file_path = os.path.join(path,f)
            if "gimbal" in file_path.lower():
                continue
            if "frame" in f and "txt" in f:
                dvs_data.frame = LoadFrameData(file_path)
                print("frame:", dvs_data.frame.shape, end="    ")
            elif "gyro" in f:
                dvs_data.gyro = LoadGyroData(file_path)
                dvs_data.gyro = preprocess_gyro(dvs_data.gyro) 
                print("gyro:", dvs_data.gyro.shape, end="    ")
            elif "ois" in f and "txt" in f:
                dvs_data.ois = LoadOISData(file_path)
                print("ois:", dvs_data.ois.shape, end="    ")
            elif f.lower() == "frames":
                dvs_data.frame_paths = self._load_frame_paths(file_path)
                print("frames:", len(dvs_data.frame_paths), end="    ")
            elif f.lower().endswith(".mp4") or f.lower().endswith(".avi"):
                dvs_data.video_path = file_path
                print("video:", os.path.basename(dvs_data.video_path), end="    ")

        print()
        dvs_data.length = dvs_data.frame.shape[0] - 1
        return dvs_data

    def _load_frame_paths(self, folder):
        frame_files = [f for f in sorted(os.listdir(folder)) if self._is_image_file(f)]
        return [os.path.join(folder, f) for f in frame_files]

    @staticmethod
    def _is_image_file(filename):
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        return os.path.splitext(filename.lower())[1] in valid_ext

    def _ensure_video_cache(self, dvs_data: DVS_data):
        if dvs_data._frame_cache is not None:
            return
        if dvs_data.video_path is None:
            raise RuntimeError("No frame directory or video file found for optical flow estimation.")
        dvs_data._frame_cache = self._load_video_frames(dvs_data.video_path)

    def _load_video_frames(self, video_path):
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise IOError(f"Unable to open video file {video_path} for optical flow estimation")
        frames = []
        success, frame = capture.read()
        while success:
            frames.append(frame)
            success, frame = capture.read()
        capture.release()
        return frames

    def _load_frame_sequence(self, dvs_data: DVS_data, start_idx: int, end_idx: int):
        frames = []
        if dvs_data.frame_paths is not None:
            for frame_idx in range(start_idx, end_idx + 1):
                frame_path = dvs_data.frame_paths[frame_idx]
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if frame is None:
                    raise IOError(f"Failed to read frame {frame_path} for optical flow estimation")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            self._ensure_video_cache(dvs_data)
            for frame_idx in range(start_idx, end_idx + 1):
                if frame_idx >= len(dvs_data._frame_cache):
                    raise IndexError("Frame index out of range while loading video frames")
                frame = dvs_data._frame_cache[frame_idx]
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames

    def generate_quaternions(self, dvs_data):
        first_id = random.randint(0, dvs_data.length - self.number_train) + 1 # skip the first frame

        sample_data = np.zeros((self.number_train, 2 * self.number_real + 1, self.unit_size), dtype=np.float32)
        sample_ois = np.zeros((self.number_train, 2), dtype=np.float32)

        sample_time = np.zeros((self.number_train+1), dtype=np.float32)
        sample_time[0] = get_timestamp(dvs_data.frame, first_id - 1)

        real_postion = np.zeros((self.number_train, 4), dtype=np.float32)

        time_start = sample_time[0]

        for i in range(self.number_train):
            sample_time[i+1] = get_timestamp(dvs_data.frame, first_id + i)
            real_postion[i] = GetGyroAtTimeStamp(dvs_data.gyro, sample_time[i+1] - self.sample_freq)
            sample_ois[i] = self.get_ois_at_timestamp(dvs_data.ois, sample_time[i+1])
            for j in range(-self.number_real, self.number_real+1):
                index = j + self.number_real
                time_stamp = sample_time[i+1] + self.sample_freq * j 
                sample_data[i, index] = self.get_data_at_timestamp(dvs_data.gyro, dvs_data.ois, time_stamp, real_postion[i])
                
        sample_data = np.reshape(sample_data, (self.number_train, (2*self.number_real+1) * self.unit_size))
        return sample_data, sample_time, first_id, real_postion, sample_ois

    def load_flo(self, idx, first_id):
        if self.flow_wrapper is None:
            raise RuntimeError("Flow wrapper has not been initialized. Call set_flow_wrapper before accessing optical flow.")

        frames = self._load_frame_sequence(self.data[idx], first_id - 1, first_id + self.number_train - 1)
        flo, flo_back = self.flow_wrapper.compute_sequence(frames)

        if self.data[idx].flo_shape is None and flo.shape[0] > 0:
            self.data[idx].flo_shape = flo[0].shape

        return flo.astype(np.float32), flo_back.astype(np.float32)

    def load_real_projections(self, idx, first_id):
        real_projections = np.zeros((self.number_train + 1, self.static_options["num_grid_rows"], 3, 3))
        for i in range(self.number_train + 1):
            frame_id = i + first_id
            metadata = GetMetadata(self.data[idx].frame, frame_id - 1)
            real_projections[i] = np.array(GetProjections(self.static_options, metadata, self.data[idx].gyro, np.zeros(self.data[idx].ois.shape), no_shutter = True))
        return real_projections

    def __getitem__(self, idx):
        inputs, times, first_id, real_postion, ois = self.generate_quaternions(self.data[idx]) 
        real_projections = self.load_real_projections(idx, first_id)
        if self.no_flo:
            flo, flo_back = 0, 0
        else:
            flo, flo_back = self.load_flo(idx, first_id)
        return inputs, times, flo, flo_back, real_projections, real_postion, ois, idx

    def __len__(self):
        return self.length

    def set_flow_wrapper(self, flow_wrapper: RAFTFlowEstimator):
        self.flow_wrapper = flow_wrapper
        return self

    def get_virtual_data(self, virtual_queue, real_queue_idx, pre_times, cur_times, time_start, batch_size, number_virtual, quat_t_1):
        # virtual_queue: [batch_size, num, 5 (timestamp, quats)]
        # eular angle, 
        # deta R angular velocity [Q't-1, Q't-2] 
        # output virtual angular velocity, x, x*dtime => detaQt
        virtual_data = np.zeros((batch_size, number_virtual, 4), dtype=np.float32)
        vt_1 = np.zeros((batch_size, 4), dtype=np.float32)
        quat_t_1 = quat_t_1.numpy()
        for i in range(batch_size):
            sample_time = cur_times[i]
            for j in range(number_virtual):
                time_stamp = sample_time - self.sample_freq * (number_virtual - j) 
                virtual_data[i, j] = get_virtual_at_timestamp(virtual_queue[i], self.data[real_queue_idx[i]].gyro, time_stamp, time_start[i], quat_t_1[i])
            vt_1[i] = get_virtual_at_timestamp(virtual_queue[i], self.data[real_queue_idx[i]].gyro, pre_times[i], time_start[i], None)
        virtual_data = np.reshape(virtual_data, (batch_size, number_virtual * 4))
        return torch.tensor(virtual_data, dtype=torch.float), torch.tensor(vt_1, dtype=torch.float)

    def update_virtual_queue(self, batch_size, virtual_queue, out, times):
        virtual_data = np.zeros((batch_size, 5))
        virtual_data[:,0] = times
        virtual_data[:, 1:] = out
        virtual_data = np.expand_dims(virtual_data, axis = 1)

        if None in virtual_queue:
            virtual_queue = virtual_data
        else:
            virtual_queue = np.concatenate((virtual_queue, virtual_data), axis = 1)
        return virtual_queue

    def random_init_virtual_queue(self, batch_size, real_postion, times):
        virtual_queue = np.zeros((batch_size, 3, 5))
        virtual_queue[:, 2, 0] = times - 0.1 * self.sample_freq
        virtual_queue[:, 1, 0] = times - 1.1 * self.sample_freq
        virtual_queue[:, 0, 0] = times - 2.1 * self.sample_freq
        for i in range(batch_size):
            quat = np.random.uniform(low=-0.06, high= 0.06, size=4) # transfer to angle # 0.05
            quat[3] = 1
            quat = quat / LA.norm(quat)
            quat = norm_quat(QuaternionProduct(real_postion[i], quat))
            virtual_queue[i, 2, 1:] = quat
            virtual_queue[i, 1, 1:] = quat
            virtual_queue[i, 0, 1:] = quat
        return virtual_queue

    def get_data_at_timestamp(self, gyro_data, ois_data, time_stamp, quat_t_1):
        quat_t = GetGyroAtTimeStamp(gyro_data, time_stamp)
        quat_dif = QuaternionProduct(quat_t, QuaternionReciprocal(quat_t_1))  
        return quat_dif

    def get_ois_at_timestamp(self, ois_data, time_stamp):
        ois_t = FindOISAtTimeStamp(ois_data, time_stamp)
        ois_t = np.array(ois_t) / self.ois_ratio
        return ois_t

def get_timestamp(frame_data, idx):
    sample_time = frame_data[idx, 0]
    metadata = GetMetadata(frame_data, idx)
    timestmap_ns = metadata["timestamp_ns"] + metadata["rs_time_ns"] * 0.5
    return timestmap_ns

def preprocess_gyro(gyro, extend = 200):
    fake_gyro = np.zeros((extend, 5))
    time_start = gyro[0,0]
    for i in range(extend):
        fake_gyro[-i-1, 0] = time_start - (gyro[i+1, 0] - time_start)
        fake_gyro[-i-1, 4] = gyro[i+1, 4]
        fake_gyro[-i-1, 1:4] = -gyro[i+1, 1:4]

    new_gyro = np.concatenate((fake_gyro, gyro), axis = 0)
    return new_gyro

    

def get_virtual_at_timestamp(virtual_queue, real_queue, time_stamp, time_start, quat_t_1 = None, sample_freq = None):
    if virtual_queue is None:
        quat_t = GetGyroAtTimeStamp(real_queue, time_stamp)
    else:
        quat_t = train_GetGyroAtTimeStamp(virtual_queue, time_stamp)
        if quat_t is None:
            quat_t = GetGyroAtTimeStamp(real_queue, time_stamp)
            
    if quat_t_1 is None:
        return quat_t
    else:
        quat_dif = QuaternionProduct(quat_t, QuaternionReciprocal(quat_t_1))  
        return quat_dif
