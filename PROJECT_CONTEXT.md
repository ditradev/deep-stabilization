# Deep Video Stabilization - Comprehensive Project Context

## 1. PROJECT OVERVIEW

### 1.1 Purpose
This is a deep learning-based video stabilization system that combines:
- **Gyroscope sensor data** for device motion tracking
- **Optical Image Stabilization (OIS)** sensor data
- **Optical flow estimation** from video frames
- **Deep neural networks** (CNN + LSTM) for learning smooth camera paths

The system predicts stabilized camera orientations (quaternions) to warp video frames into smooth, stabilized output videos.

### 1.2 Research Paper
- **Title**: "Deep Online Fused Video Stabilization"
- **Conference**: WACV 2022
- **Key Innovation**: Fuses gyroscope, OIS, and optical flow using deep learning for superior stabilization

### 1.3 Current State
- **Python Version**: 3.6
- **PyTorch Version**: 1.0.0
- **Status**: Functional but outdated, requires manual CUDA compilation
- **Main Bottleneck**: FlowNet2 dependency with custom CUDA kernels

---

## 2. ARCHITECTURE DEEP DIVE

### 2.1 Data Flow Architecture

```
Input Video (.mp4)
    ↓
[Frame Extraction] → RGB frames
    ↓
[Sensor Data Loading]
    ├── Gyroscope data (quaternions, timestamps)
    ├── OIS data (lens offsets, timestamps)
    └── Frame metadata (timestamps, rolling shutter info)
    ↓
[FlowNet2] ← **REPLACEMENT TARGET**
    ├── Forward optical flow (t → t+1)
    └── Backward optical flow (t → t-1)
    ↓
[Dataset Processing]
    ├── Temporal window: 2 seconds of data
    ├── Real gyro quaternions: t-10 to t+10 frames
    ├── Virtual quaternions: predicted stabilized path
    └── Optical flow features: 270×480×2 arrays
    ↓
[Deep Neural Network]
    ├── UNet (optical flow feature extraction)
    ├── LSTM (temporal sequence modeling)
    └── FC layers (quaternion output)
    ↓
[Quaternion Predictions] → Stabilized camera path
    ↓
[Video Warping] → Stabilized video output
```

### 2.2 Neural Network Architecture

#### Model Components (model.py)

**1. UNet (Optical Flow Processor)**
- **Input**: Optical flow [B, H, W, 4] (forward + backward flow)
- **Architecture**:
  - Input channels: 4 (2 for forward flow xy, 2 for backward flow xy)
  - Encoder: DoubleConv → Down × 4 (8→16→32→64→128 channels)
  - Pooling: MaxPool2d with stride 4
  - Global Average Pooling: Reduces spatial dimensions to 1×1
  - FC layer: 128 → 64 features
- **Output**: 64-dimensional optical flow feature vector
- **Purpose**: Compress optical flow into compact representation

**2. CNN Module (Optional, currently unused)**
- Configurable via YAML: layers, kernels, activations
- Currently `layers: null` in config → CNN is bypassed
- Would process concatenated input features if enabled

**3. LSTM Module (Temporal Processor)**
- **Input Size**: Computed dynamically based on:
  - Real quaternions: `(2×number_real + 1 + number_virtual) × 4` = 21×4 = 84 dims
  - Optical flow features: 64 dims from UNet
  - **Total**: 84 + 64 = 148 dimensions
- **Architecture**:
  - Layer 1: LSTM(148 → 512, bias=True)
  - Layer 2: LSTM(512 → 512, bias=True)
- **Hidden State**: Maintained across timesteps for sequential prediction
- **Output**: 512-dimensional temporal features

**4. Fully Connected Layers**
- **Input**: 512 dimensions from LSTM
- **Architecture**:
  - FC1: 512 → 256 (ReLU activation)
  - FC2: 256 → 4 (No activation, quaternion output)
- **Output**: Raw 4D quaternion [x, y, z, w]
- **Post-processing**: 
  - Softshrink(0.0006) applied to xyz components
  - Normalization to unit quaternion

#### Input Feature Construction

For each timestep t, the model receives:

**Real Gyro Data (84 dims)**:
- 21 quaternions spanning t-10 to t+10 (each 4 dims)
- Represents camera orientation history and near-future context
- Quaternions are relative to anchor position

**Virtual Data (8 dims)**:
- 2 virtual quaternions from previous predictions
- Provides stabilization history context

**Optical Flow Features (64 dims)**:
- Forward flow (t → t+1) and backward flow (t → t-1)
- Processed through UNet to extract motion features
- Normalized by frame dimensions (H, W)

**OIS Data (2 dims)**: 
- Lens offset percentages [x, y]
- Currently passed but not used in forward pass

### 2.3 Loss Functions (loss.py)

The training objective combines 7 weighted loss terms:

**1. Follow Loss** (weight: 10)
- **Formula**: MSE(virtual_pose, real_pose)
- **Purpose**: Ensures stabilized path follows general camera motion
- **Context**: Compares predicted virtual quaternion with 5 neighboring real quaternions (t-2 to t+2)

**2. Angle Loss** (weight: 1)
- **Formula**: Weighted angle difference with logistic activation
- **Threshold**: 6 degrees (0.1047 radians)
- **Purpose**: Penalizes large deviations between virtual and current real pose
- **Mechanism**: 
  ```
  θ = arccos(Q_virtual · Q_real^-1)
  loss = θ × sigmoid(100 × (θ - threshold))
  ```

**3. C1 Smooth Loss** (weight: 10)
- **Formula**: MSE(output_quaternion, identity_quaternion)
- **Purpose**: Encourages small incremental changes (first-order smoothness)
- **Effect**: Penalizes sudden motion in stabilized path

**4. C2 Smooth Loss** (weight: 200) - **STRONGEST**
- **Formula**: MSE(Q_t, Q_{t-1} × Q_{t-2}^-1)
- **Purpose**: Second-order smoothness (constant velocity assumption)
- **Effect**: Highly penalizes acceleration/deceleration

**5. Optical Flow Loss** (weight: 0.1)
- **Complex grid-based loss**
- **Purpose**: Ensures predicted warping is consistent with observed optical flow
- **Mechanism**:
  - Projects virtual and real grids to image space
  - Warps using optical flow
  - Measures reprojection error
  - Forward consistency: Flow(Grid_real) should match Grid_virtual
  - Backward consistency: Reverse flow should reconstruct original
- **Validation**: Clips loss to [0, 1] per sample

**6. Undefine Loss** (weight: 2.0)
- **Purpose**: Prevents undefined regions (black borders) in output
- **Mechanism**:
  - Defines safety margin (8% of frame)
  - Projects 4 corner points through virtual→real transform
  - Penalizes if corners project outside safe zone
  - Inner safety threshold: 4%
- **Effect**: Limits maximum rotation magnitude

**7. Stay Loss** (weight: 0)
- **Formula**: Mean absolute difference from identity quaternion
- **Purpose**: Encourages no rotation (for stationary camera scenarios)
- **Default**: Disabled (weight=0)

**Loss Scheduling**:
```
Epochs 1-30:   Follow + Angle + Smooth + C2_Smooth
Epochs 31-40:  Above + Undefine
Epochs 41+:    All losses including Optical + Stay
```

### 2.4 Data Processing Pipeline (dataset.py)

#### DVS_data Structure
```python
class DVS_data:
    .gyro         # [N, 5] array: [timestamp_ns, quat_x, quat_y, quat_z, quat_w]
    .ois          # [M, 3] array: [offset_x, offset_y, timestamp_ns]
    .frame        # [F, 5] array: [timestamp_ns, ..., rs_time_ns, ois_timestamp_ns]
    .flo_path     # List of paths to .flo files (forward optical flow)
    .flo_back_path # List of paths to .flo files (backward optical flow)
    .flo_shape    # (H, W) tuple, typically (270, 480)
    .length       # Number of valid frames
```

#### Dataset Parameters
- **sample_freq**: 40ms between real quaternion samples
- **number_real**: 10 (creates 21-element window: t-10 to t+10)
- **number_virtual**: 2 (uses 2 previous predictions)
- **time_train**: 2000ms per training batch
- **Frames per batch**: 2000/40 = 50 timesteps

#### Quaternion Processing
- **Coordinate System**: [x, y, z, w] convention
- **Normalization**: All quaternions normalized to unit length
- **Interpolation**: SLERP (Spherical Linear Interpolation) for timestamp alignment
- **Relative Quaternions**: 
  ```
  Δq = q_t × q_{t-1}^{-1}
  ```
  Represents incremental rotation between timesteps

#### Virtual Queue Mechanism
- **Initialization**: Random perturbation around real pose
  ```python
  random_offset = uniform(-0.06, 0.06, size=4)
  virtual_init = real_pose × normalize([random_offset..., 1])
  ```
- **Update**: Each prediction appended to queue
  ```
  virtual_queue: [timestamp, quat_x, quat_y, quat_z, quat_w]
  ```
- **Interpolation**: Uses SLERP to retrieve quaternion at arbitrary timestamp

### 2.5 FlowNet2 Integration (CRITICAL for Migration)

#### Current Usage Pattern

**1. Installation (flownet2/install.sh)**:
```bash
#!/bin/bash
# Installs 3 custom CUDA packages
cd networks/channelnorm_package && python setup.py install && cd ../..
cd networks/correlation_package && python setup.py install && cd ../..
cd networks/resample2d_package && python setup.py install && cd ../..
```

**2. Custom CUDA Operations**:
- **ChannelNorm**: Normalizes flow channels
- **Correlation**: Computes cost volume for matching
- **Resample2d**: Backward warping using flow

**3. FlowNet2 Execution (flownet2/run.sh)**:
```bash
python main.py --inference --model FlowNet2 \
  --save_flow --inference_dataset ImagesFromFolder \
  --inference_dataset_root <video_frames> \
  --resume FlowNet2_checkpoint.pth.tar
```

**4. Output Format**:
- **.flo files** (Middlebury format): Binary files storing [H, W, 2] float32 arrays
- **Forward flow**: Maps pixel (x,y) at frame_t to frame_{t+1}
- **Backward flow**: Maps pixel (x,y) at frame_t to frame_{t-1}
- **Stored in**: `video_folder/flo/` and `video_folder/flo_back/`

**5. Integration in Code**:
```python
# dataset.py
from flownet2 import flow_utils
flo = flow_utils.readFlow(flo_path)  # Returns [H, W, 2] numpy array

# inference.py / train.py
flo_out = model.unet(flo_step, flo_back_step)  # UNet processes flow
out = model.net(inputs, flo_out, ois_step)     # LSTM uses flow features
```

#### Why FlowNet2 is Problematic
1. **CUDA Compilation**: Requires gcc/nvcc compatibility (often breaks)
2. **Python 3.6 Only**: Doesn't work with modern Python
3. **PyTorch 1.0.0**: Incompatible with PyTorch 2.x
4. **Large Model**: 650MB checkpoint, slow inference
5. **Deprecated**: No longer maintained

---

## 3. CODE STRUCTURE

### 3.1 File-by-File Breakdown

#### Core Training/Inference
| File | Purpose | Key Functions | FlowNet2 Dependencies |
|------|---------|---------------|----------------------|
| `train.py` | Model training loop | `run_epoch()`, `train()` | Loads .flo files, passes to UNet |
| `inference.py` | Video stabilization | `run()`, `inference()`, `visual_result()` | Same as train.py |
| `model.py` | Neural network definitions | `Net`, `UNet`, `Model` | UNet expects flow shape [B,H,W,4] |
| `loss.py` | Loss function implementations | `Optical_loss`, `C2_Smooth_loss`, etc. | Optical_loss uses flow for consistency |
| `dataset.py` | Data loading and preprocessing | `Dataset_Gyro`, `load_flo()` | Loads .flo files via flow_utils |

#### Utilities
| File | Purpose | Dependencies |
|------|---------|--------------|
| `util.py` | Helper functions (optimizer, directories, flow normalization) | None |
| `printer.py` | Logging to console and file | None |
| `metrics.py` | Evaluation metrics (cropping, distortion, stability) | Uses warped videos |

#### Gyro Processing
| File | Purpose | Key Operations |
|------|---------|----------------|
| `gyro/gyro_function.py` | Quaternion math, projection, SLERP | 600+ lines of quaternion operations |
| `gyro/gyro_io.py` | Load sensor data from text files | Parses gyro/OIS/frame logs |
| `gyro/__init__.py` | Exports all gyro functions | - |

#### Video Warping
| File | Purpose | Technology |
|------|---------|------------|
| `warp/warping.py` | Apply stabilization to video | Mesh-based warping |
| `warp/rasterizer.py` | GPU-accelerated mesh rendering | PyTorch custom CUDA (differentiable) |
| `warp/read_write.py` | Video I/O, frame extraction | OpenCV VideoCapture/VideoWriter |

#### FlowNet2 (TO BE REMOVED)
| Directory/File | Purpose |
|----------------|---------|
| `flownet2/main.py` | Inference script for flow generation |
| `flownet2/models.py` | FlowNet2 model definition |
| `flownet2/networks/` | Custom CUDA kernels (correlation, resample, channelnorm) |
| `flownet2/install.sh` | Compiles CUDA extensions |
| `flownet2/run.sh` | Batch processes videos to generate .flo files |

### 3.2 Configuration System

#### YAML Configuration (conf/stabilzation.yaml)
```yaml
data:
  exp: 'stabilzation'                # Experiment name
  checkpoints_dir: './checkpoint'    # Model save location
  data_dir: './video'                # Input video folder
  resize_ratio: 0.25                 # Flow downsampling (1080p → 270p)
  number_real: 10                    # ±10 gyro samples
  number_virtual: 2                  # 2 virtual history samples
  time_train: 2000                   # 2 seconds per batch
  sample_freq: 40                    # 40ms sampling
  
model:
  load_model: null                   # Checkpoint path (or null)
  cnn:
    layers: null                     # CNN disabled
  rnn:
    layers:
      - [512, true]                  # LSTM layer 1
      - [512, true]                  # LSTM layer 2
  fc:
    layers:
      - [256, true]                  # FC layer 1
      - [4, true]                    # Output layer (quaternion)

train:
  optimizer: "adam"
  init_lr: 0.0001
  lr_decay: 0.5
  lr_step: 200
  epoch: 400
  clip_norm: False
  
loss:
  follow: 10
  angle: 1
  smooth: 10
  c2_smooth: 200                     # Strongest loss
  undefine: 2.0
  opt: 0.1
  stay: 0
```

### 3.3 Data Format Specifications

#### Input Video Folder Structure
```
video/
└── s_114_outdoor_running_trail_daytime/
    ├── ControlCam_20200930_104820.mp4          # Original unstabilized video
    ├── ControlCam_20200930_104820_frame.txt    # Frame metadata
    ├── ControlCam_20200930_104820_gyro.txt     # Gyroscope log
    ├── ControlCam_20200930_104820_ois.txt      # OIS log
    ├── flo/                                     # Forward optical flow
    │   ├── 00000.flo
    │   ├── 00001.flo
    │   └── ...
    └── flo_back/                                # Backward optical flow
        ├── 00000.flo
        ├── 00001.flo
        └── ...
```

#### Gyro Data Format (gyro.txt)
```
# Each line: timestamp_ns quat_x quat_y quat_z quat_w
1601468900123456789 0.001234 -0.005678 0.000123 0.999999
1601468900125456789 0.001235 -0.005679 0.000124 0.999998
...
```

#### OIS Data Format (ois.txt)
```
# Each line: offset_x offset_y timestamp_ns
0.5 -0.3 1601468900123456789
0.4 -0.2 1601468900125456789
...
```

#### Frame Metadata Format (frame.txt)
```
# Each line: timestamp_ns frame_id exposure_time_ns rs_time_ns ois_timestamp_ns
1601468900123456789 0 30000000 33000000 1601468900120000000
1601468900156456789 1 30000000 33000000 1601468900153000000
...
```

#### Optical Flow Format (.flo)
- **Binary format** (Middlebury standard)
- **Header**: Magic number (float: 202021.25)
- **Dimensions**: Width (int32), Height (int32)
- **Data**: H×W×2 float32 array (dx, dy per pixel)
- **Units**: Pixel displacement

---

## 4. DEPENDENCIES AND ENVIRONMENT

### 4.1 Current Requirements (requirements.txt)
```
colorama==0.4.4          # Terminal colors
ffmpeg==1.4              # Video processing wrapper
imageio==2.9.0           # Image I/O
matplotlib==3.3.4        # Visualization
opencv-contrib-python==4.5.1.48  # Video I/O, warping
opencv-python==4.5.1.48
pytz==2021.1             # Timezone handling
PyYAML==5.4.1            # Config parsing
scipy==1.5.4             # Scientific computing
tensorboardX==2.1        # Training visualization
tqdm==4.59.0             # Progress bars
```

**MISSING from requirements.txt**:
- `torch==1.0.0`
- `torchvision==0.2.1`
- `numpy` (assumed installed)

### 4.2 FlowNet2 Specific Requirements
- **Python**: 3.6 (strict)
- **PyTorch**: 1.0.0 (strict)
- **CUDA**: 9.0 or 10.0
- **Compiler**: gcc 5.x or 6.x (newer versions break)
- **cuDNN**: Compatible with CUDA version
- **Custom packages**: Compiled from flownet2/networks/

### 4.3 Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (11GB+ VRAM recommended)
- **RAM**: 32GB+ for training
- **Storage**: 50GB+ for dataset

---

## 5. WORKFLOWS

### 5.1 Inference Workflow (End-to-End)

**Step 1: Prepare Input Data**
```bash
# Video folder structure
./video/my_video/
  ├── video.mp4
  ├── video_frame.txt
  ├── video_gyro.txt
  └── video_ois.txt
```

**Step 2: Extract Frames and Generate Flow** (CURRENT - TO BE CHANGED)
```bash
# Extract frames
python warp/read_write.py --video_path ./video/my_video/video.mp4

# Generate optical flow using FlowNet2
cd flownet2
bash run.sh
# Creates: ./video/my_video/flo/ and ./video/my_video/flo_back/
```

**Step 3: Run Stabilization Inference**
```bash
python inference.py --config ./conf/stabilzation.yaml --dir_path ./video
```

**Internal Process**:
1. Load pretrained model from checkpoint
2. Load video data (gyro, OIS, frames, flow)
3. Initialize LSTM hidden states
4. For each timestep:
   - Construct input features (real gyro + virtual history + flow)
   - Forward pass through UNet (flow features)
   - Forward pass through LSTM (temporal modeling)
   - Predict stabilization quaternion
   - Update virtual queue
5. Save virtual quaternion trajectory

**Step 4: Warp Video**
```python
# Executed automatically within inference.py
grid = get_grid(...)  # Compute warping grid from quaternions
warp_video(grid, video_path, save_path)
```

**Output Files**:
```
./test/stabilzation/my_video/
  ├── my_video.txt                    # Virtual quaternion trajectory
  ├── my_video.jpg                    # Trajectory visualization
  └── my_video_stab.mp4              # Stabilized video (uncropped)
```

**Step 5: Evaluate Metrics** (Optional)
```bash
python metrics.py
# Creates: my_video_stab_crop.mp4 (cropped to remove borders)
# Prints: Stability, distortion, cropping metrics
```

### 5.2 Training Workflow

**Step 1: Prepare Dataset**
```bash
# Download and extract dataset
wget https://storage.googleapis.com/dataset_release/all.zip
unzip all.zip
# Creates: ./dataset_release/training/ and ./dataset_release/test/
```

**Step 2: Generate Optical Flow for All Videos**
```bash
python warp/read_write.py --dir_path ./dataset_release
cd flownet2
bash run_release.sh
```

**Step 3: Configure Training**
Edit `conf/stabilzation_train.yaml`:
```yaml
data:
  data_dir: './dataset_release'
  batch_size: 16
model:
  load_model: null  # Start from scratch
train:
  epoch: 400
  init_lr: 0.0001
```

**Step 4: Run Training**
```bash
python train.py --config ./conf/stabilzation_train.yaml
```

**Training Loop Details**:
1. Randomly sample 2-second clip from each video
2. Initialize virtual queue with random perturbation
3. For each timestep in clip:
   - Forward pass
   - Compute all 7 losses
   - Backward pass
   - Optimizer step
   - Update virtual queue for next timestep
4. Save checkpoint every 2 epochs

**Checkpoints Saved**:
```
./checkpoint/stabilzation_train/
  ├── stabilzation_train_last.checkpoint      # Latest model
  ├── stabilzation_train_epoch2.checkpoint
  ├── stabilzation_train_epoch4.checkpoint
  └── ...
```

### 5.3 Key Execution Details

#### CUDA Device Management
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Hardcoded in train.py, inference.py
```

#### LSTM Hidden State Management
- **Initialization**: `model.net.init_hidden(batch_size)` before each video
- **Persistence**: Hidden states carry across timesteps within a video
- **Reset**: New video = new initialization

#### Quaternion Accumulation
```python
# Prediction is incremental (Δq)
out = model.net(...)                          # Predict Δq
virtual_position = virtual_inputs[:, -4:]     # Previous virtual quaternion
pos = QuaternionProduct(virtual_position, anchor)  # Convert to world frame
out = QuaternionProduct(out, pos)             # Accumulate: q_t = Δq × q_{t-1}
```

---

## 6. CRITICAL MIGRATION POINTS

### 6.1 FlowNet2 Replacement Candidates

#### Option 1: RAFT (Recommended)
- **Advantage**: State-of-the-art optical flow, pure PyTorch, no CUDA compilation
- **Repository**: https://github.com/princeton-vl/RAFT
- **Installation**: `pip install` only
- **Model Size**: 15MB (vs 650MB FlowNet2)
- **Speed**: Faster than FlowNet2
- **Output**: Same format [H, W, 2]

#### Option 2: GMFlow
- **Advantage**: Even faster than RAFT, better generalization
- **Repository**: https://github.com/haofeixu/gmflow
- **Installation**: Pure PyTorch
- **Trade-off**: Slightly different flow characteristics

#### Option 3: Depth-Based Alternative (MiDaS/Depth-Anything)
- **Paradigm Shift**: Replace optical flow with depth + pose estimation
- **Advantage**: More robust to fast motion
- **Challenge**: Requires architecture changes (depth != flow)
- **Recommendation**: Phase 2 exploration, not initial replacement

### 6.2 Code Changes Required

#### Minimal Changes (RAFT/GMFlow)
1. **dataset.py**:
   ```python
   # OLD
   from flownet2 import flow_utils
   flo = flow_utils.readFlow(flo_path)
   
   # NEW
   import numpy as np
   flo = np.load(flo_path)  # Save as .npy instead of .flo
   ```

2. **flow_generation.py** (NEW FILE):
   ```python
   # Replace flownet2/run.sh with Python script
   import torch
   from raft import RAFT
   
   model = RAFT(...)
   model.load_state_dict(torch.load('raft-things.pth'))
   
   # Process video frames
   flow_forward = model(frame1, frame2)
   flow_backward = model(frame2, frame1)
   ```

3. **model.py**: No changes needed (UNet input shape unchanged)

4. **loss.py**: No changes needed (flow format identical)

#### PyTorch API Updates

**Deprecated APIs to Replace**:
```python
# OLD (PyTorch 1.0)
Variable(tensor, requires_grad=True)
nn.functional.upsample_bilinear()

# NEW (PyTorch 2.2+)
tensor.requires_grad_(True)  # No Variable wrapper needed
nn.functional.interpolate(..., mode='bilinear')
```

**Device Management**:
```python
# OLD
tensor.cuda()

# NEW (Better Practice)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor.to(device)
```

### 6.3 Testing Strategy

#### Phase 1: Verify Flow Equivalence
```python
# Generate flow with FlowNet2 (old environment)
flow_old = flownet2_inference(frame1, frame2)

# Generate flow with RAFT (new environment)
flow_new = raft_inference(frame1, frame2)

# Compare
mae = np.mean(np.abs(flow_old - flow_new))
print(f"Mean flow difference: {mae} pixels")  # Target: <1.0
```

#### Phase 2: Model Output Validation
```python
# Load old checkpoint in new environment
model_new.load_state_dict(torch.load('old_checkpoint.pth'))

# Compare predictions
quat_old = old_inference(video)
quat_new = new_inference(video)

# Angle difference
angle_diff = compute_angle_difference(quat_old, quat_new)
print(f"Mean angle difference: {angle_diff} degrees")  # Target: <0.5
```

#### Phase 3: End-to-End Validation
- Run full inference on test videos
- Compare output videos frame-by-frame
- Evaluate metrics (stability, distortion, cropping)

---

## 7. KNOWN ISSUES AND GOTCHAS

### 7.1 Current Implementation Issues

1. **Hardcoded Device**: `CUDA_VISIBLE_DEVICES` hardcoded in scripts
2. **No CPU Fallback**: Crashes if CUDA unavailable
3. **Memory Leaks**: LSTM hidden states not properly cleared in long sequences
4. **Non-deterministic Training**: No `torch.backends.cudnn.deterministic = True`
5. **YAML Security**: Uses unsafe `yaml.load()` instead of `yaml.safe_load()`

### 7.2 Data Format Assumptions

1. **Frame Rate**: Assumes 30fps (33ms frame time)
2. **Resolution**: Expects 1920×1080 input (hardcoded in gyro_function.py)
3. **Flow Resolution**: 270×480 (1/4 scale via resize_ratio=0.25)
4. **Gyro Frequency**: ~300Hz (typical smartphone IMU)
5. **OIS Frequency**: ~60Hz (typical lens actuator)

### 7.3 Quaternion Conventions

- **Coordinate System**: Right-handed, camera looking down +Z axis
- **Rotation Direction**: Right-hand rule
- **Normalization**: Critical, loss can become NaN if quaternions denormalize
- **Interpolation**: SLERP required (linear interpolation incorrect)

---

## 8. PERFORMANCE CHARACTERISTICS

### 8.1 Current Performance

**Training** (PyTorch 1.0, NVIDIA RTX 2080 Ti):
- Batch size: 16 videos
- Time per epoch: ~20 minutes
- Total training time: 400 epochs × 20min = ~130 hours

**Inference** (Single Video):
- Video length: 60 seconds, 1080p30
- FlowNet2 time: ~5 minutes
- Stabilization inference: ~1 minute
- Video warping: ~2 minutes
- **Total**: ~8 minutes

**Memory Usage**:
- Model parameters: ~12M (LSTM + UNet)
- LSTM hidden states: 512×2 = 1K floats per video
- Optical flow cache: 270×480×2×50 = 12MB per 2-second batch

### 8.2 Expected Performance After Migration

**Inference Speed Improvements**:
- RAFT flow generation: 2-3× faster than FlowNet2
- PyTorch 2.2+ optimizations: 20-30% faster training
- Compiled models (`torch.compile`): Additional 10-20% speedup

**Memory Efficiency**:
- No FlowNet2 custom kernels: -500MB VRAM
- Smaller RAFT model: -600MB VRAM
- Modern PyTorch memory management: Better utilization

---

## 9. EXPECTED OUTPUTS

### 9.1 Inference Outputs

**Stabilized Video**:
- **Format**: MP4 (H.264 codec)
- **Resolution**: Same as input (e.g., 1920×1080)
- **Frame Rate**: Same as input (e.g., 30fps)
- **Naming**: `{video_name}_stab.mp4` (uncropped), `{video_name}_stab_crop.mp4` (cropped)

**Trajectory Visualization**:
- **Format**: JPG image
- **Content**: Matplotlib plot showing:
  - Green: Original camera trajectory (gyro)
  - Blue: Stabilized trajectory (virtual)
  - X-axis: Time (frames)
  - Y-axis: Rotation angle (degrees)
  - 3 subplots: Pitch, Yaw, Roll

**Virtual Quaternion Trajectory**:
- **Format**: Text file (`.txt`)
- **Content**: Each line: `timestamp quat_x quat_y quat_z quat_w`
- **Usage**: Can be reloaded for visualization or reprocessing

### 9.2 Training Outputs

**Checkpoints**:
- **Format**: PyTorch `.checkpoint` files
- **Content**:
  ```python
  {
      'cnn': {...},
      'fc': {...},
      'state_dict': {...},  # Model weights
      'unet': {...},        # UNet weights
      'optim_dict': {...},  # Optimizer state
      'epoch': 42
  }
  ```

**Logs**:
- **Format**: Text file (`.log`)
- **Content**: Training/validation loss per epoch, timestamps, hyperparameters

**TensorBoard Logs**:
- **Directory**: `./runs/`
- **Metrics**: Loss curves, learning rate schedule

---

## 10. TESTING AND VALIDATION

### 10.1 Unit Test Targets

1. **Quaternion Operations** (gyro/gyro_function.py):
   - Test quaternion product, reciprocal, normalization
   - Verify SLERP interpolation
   - Check axis-angle conversion

2. **Data Loading** (dataset.py):
   - Test gyro/OIS data parsing
   - Verify timestamp alignment
   - Check flow loading

3. **Loss Functions** (loss.py):
   - Verify gradient flow (no NaNs)
   - Test edge cases (identity quaternion, large rotations)

4. **Model Architecture** (model.py):
   - Test input/output shapes
   - Verify LSTM state persistence
   - Check quaternion normalization

### 10.2 Integration Test Targets

1. **End-to-End Inference**:
   - Input: Sample video + sensor data
   - Output: Stabilized video
   - Validation: Visual inspection, metric thresholds

2. **Training Loop**:
   - Single epoch training
   - Checkpoint saving/loading
   - Loss convergence

3. **Flow Replacement**:
   - Compare old vs new flow outputs
   - Verify model predictions unchanged

### 10.3 Validation Metrics

**Stability** (Lower is better):
```
stability = mean(|q_t × q_{t-1}^{-1}|)
```
Measures smoothness of camera path

**Distortion** (Lower is better):
```
distortion = mean(|flow_original - flow_stabilized|)
```
Measures how much content is warped

**Cropping Ratio** (Higher is better):
```
cropping_ratio = area_valid / area_total
```
Percentage of frame retained after stabilization

**Target Values** (from paper):
- Stability: <5 degrees/second
- Cropping: >85%
- Distortion: <10 pixels

---

## 11. DOCUMENTATION AND RESOURCES

### 11.1 External Resources

- **Paper**: https://openaccess.thecvf.com/content/WACV2022/papers/Shi_Deep_Online_Fused_Video_Stabilization_WACV_2022_paper.pdf
- **Project Page**: https://zhmeishi.github.io/dvs/
- **Dataset**: https://storage.googleapis.com/dataset_release/all.zip (10GB)
- **Pretrained Model**: Included in checkpoint/stabilzation/

### 11.2 Key Academic Context

**Related Work**:
- Traditional stabilization: 2D/3D motion models, Kalman filtering
- Gyro-based: Use IMU for camera path, warp frames
- Deep learning: Learn from data, end-to-end optimization

**This Method's Novelty**:
- **Fusion**: Combines gyro, OIS, and optical flow
- **Online**: Real-time capable (processes sequentially)
- **Deep**: Learns optimal fusion and smoothing from data

**Limitations Mentioned in Paper**:
- Fails on very fast motion (motion blur)
- Requires calibrated sensors (gyro-camera sync)
- Training data bias (outdoor daytime videos)

---

## 12. MIGRATION PREREQUISITES

### 12.1 Knowledge Requirements

**Must Understand**:
- PyTorch basics (tensors, modules, optimizers)
- Quaternion mathematics (rotation representation)
- Optical flow concepts (motion estimation)
- Video processing (frame extraction, codec)

**Should Understand**:
- LSTM/RNN architectures
- Multi-term loss functions
- Sensor fusion concepts
- Homography transforms

**Nice to Have**:
- FlowNet2 architecture
- Rolling shutter effects
- Camera calibration
- GPU programming

### 12.2 Tools Required

**Essential**:
- Python 3.10+ environment
- PyTorch 2.2+ with CUDA
- Git for version control
- CUDA-capable GPU (development)

**Recommended**:
- Conda/virtualenv for environment isolation
- TensorBoard for training monitoring
- FFmpeg for video processing
- Jupyter for experimentation

### 12.3 Time Estimates

**Minimum Viable Migration** (RAFT replacement only):
- Environment setup: 2 hours
- Flow generation script: 4 hours
- API updates: 4 hours
- Testing: 8 hours
- **Total**: 2-3 days

**Complete Migration** (with refactoring):
- Above + code cleanup: 3 days
- Documentation: 2 days
- Comprehensive testing: 3 days
- **Total**: 1-2 weeks

**Exploratory Migration** (Depth-Anything alternative):
- Research phase: 1 week
- Architecture changes: 1 week
- Retraining: 1 week
- **Total**: 3-4 weeks

---

## 13. SUCCESS CRITERIA

### 13.1 Functional Requirements

✅ **Installation**: `pip install -r requirements.txt` works on Python 3.10+

✅ **Inference**: Successfully stabilizes test video

✅ **Training**: Loss converges without errors

✅ **Output Format**: Identical file naming and structure

✅ **No Manual Steps**: No CUDA compilation or shell scripts

### 13.2 Performance Requirements

✅ **Speed**: Inference time ≤ current implementation

✅ **Quality**: Stabilization metrics within 5% of original

✅ **Memory**: Peak VRAM usage ≤ current implementation

✅ **Accuracy**: Quaternion predictions MAE <0.01 vs old model

### 13.3 Code Quality Requirements

✅ **Python 3.10+**: No deprecated syntax

✅ **PyTorch 2.2+**: No deprecated APIs

✅ **Type Hints**: Added to public functions

✅ **Documentation**: README updated with new instructions

✅ **Error Handling**: Graceful failure messages

---

## 14. EDGE CASES AND FAILURE MODES

### 14.1 Known Failure Scenarios

1. **Extreme Rotation**: >90° rotation in single frame
   - **Symptom**: Undefine loss spikes, NaN in output
   - **Cause**: Projection goes out of bounds
   - **Mitigation**: Clipping loss to [0, 1]

2. **Sensor Desync**: Gyro timestamp misalignment >100ms
   - **Symptom**: Jittery output, high optical loss
   - **Cause**: Quaternion interpolation incorrect
   - **Mitigation**: Validate timestamps in dataset loader

3. **Motion Blur**: Fast panning or low light
   - **Symptom**: Flow estimation fails, stabilization disabled
   - **Cause**: Optical flow unreliable
   - **Mitigation**: Fall back to gyro-only mode

4. **LSTM Divergence**: Rare training instability
   - **Symptom**: Loss explodes after many epochs
   - **Cause**: Gradient explosion in long sequences
   - **Mitigation**: Gradient clipping (clip_norm parameter)

### 14.2 Data Quality Requirements

**Minimum Requirements**:
- Gyro frequency: ≥100Hz
- Frame rate: ≥24fps
- Gyro-camera time sync: ±10ms
- Video resolution: ≥720p

**Recommended**:
- Gyro frequency: ≥200Hz
- Frame rate: 30fps
- Time sync: ±5ms
- Resolution: 1080p

---

## 15. CONTEXT FOR AI AGENTS

### 15.1 Code Reading Strategy

**Start Here** (Understand core logic):
1. `inference.py`: Main execution flow
2. `model.py`: Neural network architecture
3. `loss.py`: Training objectives

**Then Explore** (Dependencies):
4. `dataset.py`: Data loading and preprocessing
5. `gyro/gyro_function.py`: Quaternion operations
6. `warp/warping.py`: Output generation

**Finally Review** (Supporting code):
7. `util.py`, `printer.py`, `metrics.py`
8. Config files: `conf/*.yaml`

### 15.2 Code Modification Strategy

**Safe to Change** (No side effects):
- Requirements versions
- Device management (cuda() → to(device))
- Deprecated API calls
- Print statements and logging

**Change with Caution** (Test thoroughly):
- Loss function weights
- Model architecture
- Data loader logic
- Quaternion operations

**Do Not Change** (Unless necessary):
- Quaternion math fundamentals
- Coordinate system conventions
- File format parsers
- Loss function logic

### 15.3 Testing Strategy for AI

**After Each Change**:
```python
# Minimal validation
python -c "import torch; from model import Model; print('OK')"
```

**After Flow Replacement**:
```python
# Generate flow for test video
python generate_flow_raft.py --video test_video.mp4

# Run inference
python inference.py --config conf/stabilzation.yaml --dir_path test_video/
```

**Before Declaring Success**:
```python
# Full validation
python inference.py --config conf/stabilzation.yaml --dir_path video/
python metrics.py
# Manually inspect output video quality
```

---

## END OF PROJECT CONTEXT

This document provides a comprehensive understanding of the Deep Video Stabilization codebase. Use it as a reference when:
- Understanding code behavior
- Planning modifications
- Debugging issues
- Validating changes

For migration-specific instructions, refer to `MIGRATION_REQUIREMENTS.md`.
For step-by-step task breakdown, refer to `TASK_CHECKLIST.md`.
