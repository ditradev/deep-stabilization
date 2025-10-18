# Deep Video Stabilization - Migration Requirements

## DOCUMENT PURPOSE

This document specifies the exact requirements for migrating the Deep Video Stabilization project from its current state (Python 3.6, PyTorch 1.0, FlowNet2) to a modern environment (Python 3.10+, PyTorch 2.2+, RAFT/modern flow model). 

**Audience**: AI agents, developers, or engineers executing the migration task.

---

## MIGRATION GOALS

### Primary Goals

1. **✅ G1: Eliminate Manual Compilation**
   - Remove all custom CUDA extensions
   - Remove FlowNet2 and its dependencies
   - Replace with pure Python/PyTorch solution

2. **✅ G2: Modernize Python and PyTorch**
   - Upgrade to Python 3.10 or 3.11
   - Upgrade to PyTorch 2.2 or latest stable
   - Update all deprecated API calls

3. **✅ G3: Simplify Installation**
   - Single command: `pip install -r requirements.txt`
   - No shell scripts, no CUDA compilation
   - Cross-platform compatibility (Windows, Linux, macOS)

4. **✅ G4: Preserve Functionality**
   - Inference works identically
   - Training produces equivalent results
   - Output file naming unchanged
   - Metrics remain comparable

5. **✅ G5: Improve Documentation**
   - Clear installation instructions
   - Updated README with new workflow
   - Explanation of changes made
   - Performance comparison notes

### Secondary Goals (Nice-to-Have)

1. **⭐ G6: Performance Improvement**
   - Faster optical flow generation
   - Reduced memory footprint
   - Utilize PyTorch 2.x features (compile, better device management)

2. **⭐ G7: Code Quality**
   - Add type hints to main functions
   - Improve error handling
   - Remove hardcoded device strings
   - Add CPU fallback mode

3. **⭐ G8: Testing Infrastructure**
   - Add unit tests for quaternion operations
   - Add integration test for inference
   - Add CI/CD configuration

---

## DETAILED REQUIREMENTS

### R1: Environment and Dependencies

#### R1.1: Python Version
**Requirement**: Python 3.10 or 3.11 ONLY (3.12+ may have compatibility issues)

**Rationale**: 
- Python 3.10 is LTS and widely supported
- PyTorch 2.2 has excellent support
- Avoids cutting-edge Python features that may break dependencies

**Verification**:
```bash
python --version  # Should output: Python 3.10.x or 3.11.x
```

#### R1.2: PyTorch Version
**Requirement**: PyTorch 2.2 or 2.3 with CUDA 11.8 or 12.1

**Rationale**:
- PyTorch 2.x has major performance improvements
- CUDA 11.8/12.1 have broad GPU support
- Avoids deprecated API warnings

**Verification**:
```python
import torch
print(torch.__version__)  # Should be >= 2.2.0
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)  # Should be 11.8 or 12.1
```

#### R1.3: Updated requirements.txt
**Requirement**: Create new requirements.txt with:

```
# Core deep learning
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24.0,<2.0.0

# Optical flow model (choose ONE)
# Option 1: RAFT (recommended)
git+https://github.com/princeton-vl/RAFT.git

# Option 2: OR use this if pip package available
# raft-pytorch>=1.0.0

# Video and image processing
opencv-python>=4.8.0
imageio>=2.31.0
imageio-ffmpeg>=0.4.9

# Scientific computing
scipy>=1.10.0

# Configuration and logging
PyYAML>=6.0
tensorboard>=2.14.0
tqdm>=4.65.0

# Utilities
matplotlib>=3.7.0
pillow>=10.0.0
```

**Notes**:
- Remove `ffmpeg==1.4` (replaced by imageio-ffmpeg)
- Remove `colorama` (not essential)
- Remove `pytz` (not used)
- Remove `tensorboardX` (replaced by official tensorboard)
- Remove `opencv-contrib-python` (standard opencv-python sufficient)

**Verification**:
```bash
pip install -r requirements.txt
python -c "import torch, cv2, imageio, yaml, scipy, tqdm; print('All imports successful')"
```

### R2: FlowNet2 Replacement

#### R2.1: Choose Replacement Model

**Option 1: RAFT (RECOMMENDED)**

**Specification**:
- **Repository**: https://github.com/princeton-vl/RAFT
- **Paper**: "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow" (ECCV 2020)
- **Model Variant**: RAFT-things (trained on FlyingThings3D)
- **Checkpoint**: `raft-things.pth` (download from repo)
- **Input**: RGB frames (H×W×3), uint8 or float32
- **Output**: Flow (H×W×2), float32 numpy array

**Advantages**:
- ✅ Pure PyTorch (no CUDA compilation)
- ✅ State-of-the-art accuracy
- ✅ Faster than FlowNet2
- ✅ Smaller model (15MB vs 650MB)
- ✅ Well-maintained repository
- ✅ Easy to integrate

**Installation**:
```bash
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
pip install -r requirements.txt
# Download checkpoint
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth
```

**Integration Code Template**:
```python
import sys
sys.path.append('RAFT/core')
import torch
from raft import RAFT
from utils.utils import InputPadder
import numpy as np

class RAFTFlowEstimator:
    def __init__(self, model_path='raft-things.pth', device='cuda'):
        self.device = device
        self.model = RAFT(args=...)  # Configure args
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def estimate_flow(self, frame1, frame2):
        """
        Args:
            frame1, frame2: [H, W, 3] numpy arrays (uint8) or torch tensors
        Returns:
            flow: [H, W, 2] numpy array (float32)
        """
        # Convert to torch tensors
        if isinstance(frame1, np.ndarray):
            frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float()
            frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float()
        
        # Add batch dimension
        frame1 = frame1.unsqueeze(0).to(self.device)
        frame2 = frame2.unsqueeze(0).to(self.device)
        
        # Pad images to multiple of 8
        padder = InputPadder(frame1.shape)
        frame1, frame2 = padder.pad(frame1, frame2)
        
        # Estimate flow
        _, flow = self.model(frame1, frame2, iters=20, test_mode=True)
        flow = padder.unpad(flow)
        
        # Convert to numpy
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        return flow
```

**Option 2: GMFlow (ALTERNATIVE)**

**Specification**:
- **Repository**: https://github.com/haofeixu/gmflow
- **Advantages**: Faster than RAFT, better on long-range motion
- **Trade-offs**: Less mature, different flow characteristics
- **Use Case**: If RAFT is too slow for your hardware

**Option 3: Depth-Anything (FUTURE EXPLORATION)**

**Specification**:
- **Paradigm Change**: Replaces optical flow with monocular depth
- **Advantages**: More robust to fast motion and blur
- **Challenges**: 
  - Requires architecture changes (depth != flow)
  - Need to retrain model
  - Loss functions need adaptation
- **Recommendation**: Phase 2 after basic migration is stable

#### R2.2: Flow Generation Script

**Requirement**: Create `generate_flow.py` to replace `flownet2/run.sh`

**Specification**:
```python
#!/usr/bin/env python3
"""
Generate optical flow for video dataset.
Replaces: flownet2/run.sh

Usage:
    python generate_flow.py --data_dir ./video
    python generate_flow.py --data_dir ./dataset_release --batch_size 4
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch

# Import RAFT or chosen flow model
sys.path.append('RAFT/core')
from raft import RAFT
from utils.utils import InputPadder

def load_frames(video_dir):
    """Load all frames from video directory."""
    frame_dir = Path(video_dir) / 'frames'
    frame_files = sorted(frame_dir.glob('*.png'))  # Assuming frames extracted as PNG
    frames = []
    for fpath in frame_files:
        frame = cv2.imread(str(fpath))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames

def estimate_flows(frames, flow_estimator, output_dir_fwd, output_dir_bwd):
    """Generate forward and backward flows for all frame pairs."""
    output_dir_fwd.mkdir(parents=True, exist_ok=True)
    output_dir_bwd.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(len(frames) - 1), desc="Generating flows"):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        
        # Forward flow: t -> t+1
        flow_fwd = flow_estimator.estimate_flow(frame1, frame2)
        np.save(output_dir_fwd / f'{i:05d}.npy', flow_fwd)
        
        # Backward flow: t+1 -> t
        flow_bwd = flow_estimator.estimate_flow(frame2, frame1)
        np.save(output_dir_bwd / f'{i:05d}.npy', flow_bwd)

def process_video_folder(video_dir, flow_estimator):
    """Process a single video folder."""
    video_dir = Path(video_dir)
    print(f"Processing: {video_dir}")
    
    # Load frames
    frames = load_frames(video_dir)
    print(f"Loaded {len(frames)} frames")
    
    # Generate flows
    output_fwd = video_dir / 'flo'
    output_bwd = video_dir / 'flo_back'
    estimate_flows(frames, flow_estimator, output_fwd, output_bwd)
    print(f"Saved flows to {output_fwd} and {output_bwd}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--model', default='raft-things.pth', help='Path to RAFT checkpoint')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (future use)')
    args = parser.parse_args()
    
    # Initialize flow estimator
    print("Loading RAFT model...")
    flow_estimator = RAFTFlowEstimator(model_path=args.model, device=args.device)
    
    # Find all video folders
    data_dir = Path(args.data_dir)
    if (data_dir / 'training').exists():
        video_dirs = list((data_dir / 'training').iterdir())
        video_dirs += list((data_dir / 'test').iterdir())
    else:
        video_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    # Process each video
    for video_dir in video_dirs:
        if (video_dir / 'frames').exists():
            process_video_folder(video_dir, flow_estimator)
        else:
            print(f"Skipping {video_dir} (no frames folder)")

if __name__ == '__main__':
    main()
```

**Notes**:
- Change flow storage from `.flo` binary format to `.npy` NumPy format
- `.npy` is more Python-friendly and equally efficient
- Keep same directory structure: `video_folder/flo/` and `video_folder/flo_back/`

#### R2.3: Update Flow Loading in dataset.py

**Requirement**: Modify `dataset.py` to load `.npy` instead of `.flo`

**Change Specification**:

```python
# OLD CODE (lines 286-291)
from flownet2 import flow_utils

def LoadFlow(path):
    file_names = sorted(os.listdir(path))
    file_path =[]
    for n in file_names:
        file_path.append(os.path.join(path, n))
    return file_path, flow_utils.readFlow(file_path[0]).shape

# NEW CODE
def LoadFlow(path):
    """Load optical flow file paths and shape.
    
    Args:
        path: Directory containing .npy flow files
    
    Returns:
        file_path: List of paths to flow files
        shape: (H, W) tuple of flow dimensions
    """
    file_names = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
    file_path = [os.path.join(path, n) for n in file_names]
    
    if len(file_path) == 0:
        raise ValueError(f"No .npy files found in {path}")
    
    # Load first flow to get shape
    first_flow = np.load(file_path[0])
    shape = first_flow.shape[:2]  # (H, W)
    
    return file_path, shape
```

**And update flow reading** (lines 187-191):

```python
# OLD CODE
for i in range(self.number_train):
    frame_id = i + first_id
    f = flow_utils.readFlow(self.data[idx].flo_path[frame_id-1]).astype(np.float32) 
    flo[i] = f

# NEW CODE
for i in range(self.number_train):
    frame_id = i + first_id
    f = np.load(self.data[idx].flo_path[frame_id-1]).astype(np.float32)
    flo[i] = f
```

**Remove FlowNet2 import**:
```python
# DELETE THIS LINE (line 22)
from flownet2 import flow_utils
```

### R3: PyTorch API Updates

#### R3.1: Remove Variable Wrapper

**Deprecated API**: `from torch.autograd import Variable`

**Locations to Update**:
1. `model.py` line 6: Remove import
2. `model.py` lines 37, 79, 117, 175, 206, 406: Remove Variable() wrapper
3. `loss.py` line 3: Remove import

**Change Pattern**:
```python
# OLD
from torch.autograd import Variable
quat_out = Variable(torch.zeros((batch_size, 4), requires_grad=True))
if USE_CUDA:
    quat_out = quat_out.cuda()

# NEW
quat_out = torch.zeros((batch_size, 4), requires_grad=True)
if USE_CUDA:
    quat_out = quat_out.cuda()
```

**Files to Modify**:
- `model.py`: 6 locations
- `loss.py`: 0 locations (not using Variable)
- `gyro/gyro_function.py`: 7 locations

#### R3.2: Replace upsample_bilinear

**Deprecated API**: `torch.nn.functional.upsample_bilinear`

**Location**: `loss.py` lines 86, 89

**Change Pattern**:
```python
# OLD
grid_t = torch.nn.functional.upsample_bilinear(grid_t, size=(h, w))

# NEW
grid_t = torch.nn.functional.interpolate(grid_t, size=(h, w), mode='bilinear', align_corners=False)
```

**Notes**:
- `align_corners=False` is recommended default
- Behavior should be identical for this use case

#### R3.3: Improve Device Management

**Current Issue**: Hardcoded `.cuda()` calls throughout codebase

**Requirement**: Replace with flexible device management

**Change Pattern**:
```python
# At top of file (inference.py, train.py)
# OLD
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
USE_CUDA = cf['data']["use_cuda"]

# NEW
import torch

def get_device(config):
    """Get device based on config and availability."""
    if config['data'].get("use_cuda", True) and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

device = get_device(cf)
```

**Replace all occurrences**:
```python
# OLD
tensor.cuda()

# NEW
tensor.to(device)
```

**Files with `.cuda()` calls** (search for `.cuda\(\)` regex):
- `model.py`: 10+ occurrences
- `train.py`: 15+ occurrences
- `inference.py`: 15+ occurrences
- `loss.py`: 20+ occurrences
- `gyro/gyro_function.py`: 5+ occurrences

**Note**: This is a large refactor, prioritize for secondary pass after basic migration works

#### R3.4: Update YAML Loading

**Security Issue**: `yaml.load()` is unsafe

**Location**: `train.py` line 153, `inference.py` line 190

**Change Pattern**:
```python
# OLD
cf = yaml.load(open(config_file, 'r'))

# NEW
with open(config_file, 'r') as f:
    cf = yaml.safe_load(f)
```

**Rationale**: Prevents arbitrary code execution vulnerability

#### R3.5: Fix Deprecated Tensor Operations

**Issue**: Some tensor operations have deprecated syntax in PyTorch 2.x

**Specific Changes**:

1. **BatchNorm inplace** (model.py line 36):
```python
# OLD
self.activation = activation_function(inplace=True)

# NEW
self.activation = activation_function()  # inplace not needed for most cases
# OR keep for performance
self.activation = activation_function(inplace=True)  # Still works, just warns
```

2. **Clip grad norm** (train.py line 141):
```python
# Already using correct API, but verify:
nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=clip_norm)
# This is correct for PyTorch 2.x
```

### R4: Remove FlowNet2 Code

#### R4.1: Delete FlowNet2 Directory

**Requirement**: Remove entire `flownet2/` directory

**Command**:
```bash
rm -rf dvs/flownet2/
```

**Justification**: No longer needed after RAFT integration

#### R4.2: Remove FlowNet2 References

**Requirement**: Search and remove all FlowNet2 imports and references

**Locations**:
1. `dataset.py` line 22: `from flownet2 import flow_utils`
2. `README.md` lines 26-37: FlowNet2 Preparation section
3. Any shell scripts: `flownet2/install.sh`, `flownet2/run.sh`

**Verification**:
```bash
grep -r "flownet2" --include="*.py" --include="*.md" dvs/
# Should return no results
```

### R5: Testing and Validation

#### R5.1: Unit Tests

**Requirement**: Create `tests/test_migration.py` to verify basic functionality

**Test Specification**:
```python
import torch
import numpy as np
import sys
sys.path.append('dvs')

def test_imports():
    """Test all core imports work."""
    from model import Model, Net, UNet
    from loss import C1_Smooth_loss, Optical_loss
    from dataset import Dataset_Gyro
    from gyro import torch_norm_quat, torch_QuaternionProduct
    print("✅ All imports successful")

def test_model_creation():
    """Test model can be instantiated."""
    import yaml
    with open('dvs/conf/stabilzation.yaml', 'r') as f:
        cf = yaml.safe_load(f)
    
    from model import Model
    model = Model(cf)
    print(f"✅ Model created with {sum(p.numel() for p in model.net.parameters())} parameters")

def test_quaternion_operations():
    """Test quaternion math is correct."""
    from gyro import torch_norm_quat, torch_QuaternionProduct, torch_QuaternionReciprocal
    
    q1 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # Identity
    q2 = torch.tensor([[0.1, 0.0, 0.0, 0.995]])  # Small rotation
    
    # Test normalization
    q_norm = torch_norm_quat(q2)
    assert torch.allclose(torch.norm(q_norm[0]), torch.tensor(1.0), atol=1e-5)
    
    # Test product
    q3 = torch_QuaternionProduct(q1, q2)
    assert torch.allclose(q3, q2, atol=1e-5)
    
    # Test reciprocal
    q_inv = torch_QuaternionReciprocal(q2)
    q_identity = torch_QuaternionProduct(q2, q_inv)
    assert torch.allclose(q_identity, q1, atol=1e-4)
    
    print("✅ Quaternion operations correct")

def test_flow_loading():
    """Test flow loading works with .npy format."""
    # Create dummy flow
    dummy_flow = np.random.randn(270, 480, 2).astype(np.float32)
    np.save('/tmp/test_flow.npy', dummy_flow)
    
    # Load it back
    loaded_flow = np.load('/tmp/test_flow.npy')
    assert loaded_flow.shape == (270, 480, 2)
    assert np.allclose(loaded_flow, dummy_flow)
    
    print("✅ Flow loading works")

def test_model_forward():
    """Test model forward pass."""
    import yaml
    with open('dvs/conf/stabilzation.yaml', 'r') as f:
        cf = yaml.safe_load(f)
    
    from model import Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(cf)
    model.net.to(device)
    model.unet.to(device)
    model.net.eval()
    model.unet.eval()
    
    # Create dummy inputs
    batch_size = 2
    model.net.init_hidden(batch_size)
    
    # Input dimensions from config
    input_dim = (2*10 + 1 + 2) * 4  # 84
    inputs = torch.randn(batch_size, input_dim).to(device)
    
    # Flow input
    flo = torch.randn(batch_size, 270, 480, 2).to(device)
    flo_back = torch.randn(batch_size, 270, 480, 2).to(device)
    flo_out = model.unet(flo, flo_back)
    
    # OIS input
    ois = torch.randn(batch_size, 2).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model.net(inputs, flo_out, ois)
    
    assert output.shape == (batch_size, 4)
    assert torch.allclose(torch.norm(output, dim=1), torch.ones(batch_size), atol=1e-4)
    print("✅ Model forward pass successful")

if __name__ == '__main__':
    test_imports()
    test_model_creation()
    test_quaternion_operations()
    test_flow_loading()
    test_model_forward()
    print("\n🎉 All tests passed!")
```

**Verification**:
```bash
python tests/test_migration.py
```

#### R5.2: Integration Test - Inference

**Requirement**: Verify inference works end-to-end

**Test Specification**:
1. Use provided sample video from `video/s_114_outdoor_running_trail_daytime/`
2. Run full inference pipeline
3. Verify output files are created
4. Check output video plays correctly

**Test Script** (`tests/test_inference.py`):
```python
import os
import subprocess
import sys
from pathlib import Path

def test_inference():
    """Test full inference pipeline."""
    
    # Check data exists
    video_dir = Path('dvs/video/s_114_outdoor_running_trail_daytime')
    assert video_dir.exists(), f"Test data not found at {video_dir}"
    
    # Check flow exists (should be generated by generate_flow.py)
    flo_dir = video_dir / 'flo'
    assert flo_dir.exists(), "Flow directory not found. Run generate_flow.py first"
    assert len(list(flo_dir.glob('*.npy'))) > 0, "No flow files found"
    
    # Run inference
    cmd = [
        sys.executable, 'dvs/inference.py',
        '--config', 'dvs/conf/stabilzation.yaml',
        '--dir_path', 'dvs/video'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0, f"Inference failed with code {result.returncode}"
    
    # Check outputs
    output_dir = Path('dvs/test/stabilzation/s_114_outdoor_running_trail_daytime')
    assert output_dir.exists(), "Output directory not created"
    
    output_video = output_dir / 's_114_outdoor_running_trail_daytime_stab.mp4'
    assert output_video.exists(), "Output video not created"
    
    output_traj = output_dir / 's_114_outdoor_running_trail_daytime.txt'
    assert output_traj.exists(), "Trajectory file not created"
    
    output_plot = output_dir / 's_114_outdoor_running_trail_daytime.jpg'
    assert output_plot.exists(), "Plot file not created"
    
    print("✅ Inference test passed")

if __name__ == '__main__':
    test_inference()
```

**Verification**:
```bash
python tests/test_inference.py
```

#### R5.3: Comparison Test - Flow Quality

**Requirement**: Compare RAFT flow vs FlowNet2 flow (if old data available)

**Test Specification**:
```python
import numpy as np
import matplotlib.pyplot as plt

def compare_flows(flow_old_path, flow_new_path):
    """Compare old FlowNet2 flow with new RAFT flow."""
    
    # Load flows
    flow_old = load_flo_file(flow_old_path)  # If .flo available
    flow_new = np.load(flow_new_path)
    
    # Check shapes match
    assert flow_old.shape == flow_new.shape
    
    # Compute metrics
    mae = np.mean(np.abs(flow_old - flow_new))
    rmse = np.sqrt(np.mean((flow_old - flow_new)**2))
    
    print(f"Mean Absolute Error: {mae:.3f} pixels")
    print(f"RMSE: {rmse:.3f} pixels")
    
    # Visualize difference
    flow_diff = np.linalg.norm(flow_old - flow_new, axis=2)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(np.linalg.norm(flow_old, axis=2))
    plt.title("FlowNet2 Flow Magnitude")
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(np.linalg.norm(flow_new, axis=2))
    plt.title("RAFT Flow Magnitude")
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(flow_diff)
    plt.title("Difference Magnitude")
    plt.colorbar()
    
    plt.savefig('flow_comparison.png')
    print("Saved comparison plot to flow_comparison.png")
    
    # Acceptance criteria
    if mae < 2.0:  # Less than 2 pixels difference on average
        print("✅ Flow quality acceptable")
    else:
        print("⚠️ Flow quality degraded, review needed")
```

**Note**: This test is optional and requires access to old FlowNet2 outputs

#### R5.4: Comparison Test - Model Output

**Requirement**: Verify model predictions are unchanged

**Test Specification**:
```python
import torch
import numpy as np

def test_checkpoint_compatibility():
    """Test that old checkpoint loads in new environment."""
    
    import yaml
    with open('dvs/conf/stabilzation.yaml', 'r') as f:
        cf = yaml.safe_load(f)
    
    from model import Model
    model = Model(cf)
    
    # Load old checkpoint
    checkpoint_path = 'dvs/checkpoint/stabilzation/stabilzation_last.checkpoint'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to load state dict
    try:
        model.net.load_state_dict(checkpoint['state_dict'])
        model.unet.load_state_dict(checkpoint['unet'])
        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Checkpoint loading failed: {e}")
        raise
    
    # Test inference with loaded model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.net.to(device)
    model.unet.to(device)
    model.net.eval()
    model.unet.eval()
    
    # Run forward pass
    batch_size = 1
    model.net.init_hidden(batch_size)
    
    input_dim = (2*10 + 1 + 2) * 4
    inputs = torch.randn(batch_size, input_dim).to(device)
    flo = torch.randn(batch_size, 270, 480, 2).to(device)
    flo_back = torch.randn(batch_size, 270, 480, 2).to(device)
    flo_out = model.unet(flo, flo_back)
    ois = torch.randn(batch_size, 2).to(device)
    
    with torch.no_grad():
        output = model.net(inputs, flo_out, ois)
    
    assert output.shape == (batch_size, 4)
    print("✅ Model inference successful with loaded checkpoint")

if __name__ == '__main__':
    test_checkpoint_compatibility()
```

### R6: Documentation Updates

#### R6.1: Update README.md

**Requirement**: Rewrite README.md sections

**Sections to Update**:

1. **Environment Setting** (lines 7-14):
```markdown
## Environment Setup

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support (11.8 or 12.1)
- 16GB+ GPU VRAM (for training), 8GB+ (for inference)

### Installation
```bash
# Create virtual environment (recommended)
conda create -n dvs python=3.10
conda activate dvs

# Install dependencies
cd dvs
pip install -r requirements.txt

# Download RAFT model
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth
mv raft-things.pth ./raft_checkpoint/
```

**Installation complete!** No compilation required.
```

2. **Update Data Preparation** (lines 16-24):
```markdown
## Data Preparation

Download sample video [here](https://drive.google.com/file/d/1PpF3-6BbQKy9fldjIfwa5AlbtQflx3sG/view?usp=sharing).
Extract the `video` folder under the `dvs` folder.

Extract frames and generate optical flow:
```bash
# Extract frames from video
python warp/read_write.py --video_folder ./video/s_114_outdoor_running_trail_daytime

# Generate optical flow using RAFT
python generate_flow.py --data_dir ./video --model raft-things.pth
```

Demo of curve visualization:
The **gyro/OIS curve visualization** can be found at `dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820_real.jpg`.
```

3. **Remove FlowNet2 Preparation Section** (lines 26-37):
```markdown
## ~~FlowNet2 Preparation~~

**This section is OBSOLETE.** The project now uses RAFT for optical flow, which requires no custom CUDA compilation.
```

4. **Update Running Inference** (lines 39-50):
```markdown
## Running Inference

```bash
python inference.py --config ./conf/stabilzation.yaml --dir_path ./video
python metrics.py
```

The loss and metric information will be printed in the terminal.

**Output files** (in `dvs/test/stabilzation/`):
- `s_114_outdoor_running_trail_daytime.jpg`: Trajectory plot (blue=stabilized, green=original)
- `s_114_outdoor_running_trail_daytime_stab.mp4`: Uncropped stabilized video
- `s_114_outdoor_running_trail_daytime_stab_crop.mp4`: Cropped stabilized video (generated after running metrics.py)
```

5. **Update Training Section** (lines 52-67):
```markdown
## Training

Download dataset for training and testing [here](https://storage.googleapis.com/dataset_release/all.zip).
Extract `all.zip` and move the `dataset_release` folder under the `dvs` folder.

Generate optical flow for the dataset:
```bash
# Extract frames for all videos
python warp/read_write.py --dir_path ./dataset_release

# Generate optical flow using RAFT
python generate_flow.py --data_dir ./dataset_release
```

Run training:
```bash
python train.py --config ./conf/stabilzation_train.yaml
```

The model checkpoints are saved in `checkpoint/stabilzation_train/`.

**Training time**: ~20 minutes per epoch on RTX 3090, ~130 hours total for 400 epochs.
```

6. **Add Migration Notes Section**:
```markdown
## Migration Notes (March 2024)

This codebase has been migrated from Python 3.6 + PyTorch 1.0 + FlowNet2 to Python 3.10 + PyTorch 2.2 + RAFT.

### Key Changes
- **Optical Flow**: FlowNet2 → RAFT (15MB model, no CUDA compilation)
- **Python**: 3.6 → 3.10/3.11
- **PyTorch**: 1.0.0 → 2.2+
- **Installation**: No more custom CUDA builds, single `pip install` command
- **Performance**: ~2-3× faster flow generation, same stabilization quality

### Compatibility
- Old checkpoints (`.checkpoint` files) are compatible with the new environment
- Flow files are now saved as `.npy` (NumPy) instead of `.flo` (Middlebury format)

### Known Issues
- RAFT requires CUDA for acceptable speed (CPU inference is 10× slower)
- First inference run downloads RAFT model weights (~15MB)

For detailed migration information, see `MIGRATION_REQUIREMENTS.md`.
```

#### R6.2: Create CHANGELOG.md

**Requirement**: Document all changes made during migration

**Specification**:
```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2024-03-XX - Migration to Modern Stack

### Major Changes

#### Removed
- **FlowNet2 dependency**: Removed entire `flownet2/` directory and custom CUDA extensions
  - `flownet2/networks/channelnorm_package/`
  - `flownet2/networks/correlation_package/`
  - `flownet2/networks/resample2d_package/`
  - `flownet2/install.sh`, `flownet2/run.sh`
- **Python 3.6 requirement**: Now requires Python 3.10 or 3.11
- **PyTorch 1.0**: Upgraded to PyTorch 2.2+
- **Manual CUDA compilation**: No longer needed

#### Added
- **RAFT optical flow model**: State-of-the-art, pure PyTorch implementation
  - Model: `raft-things.pth` (15MB)
  - No compilation required
  - Faster inference than FlowNet2
- **generate_flow.py**: Python script to replace shell-based flow generation
- **Test suite**: `tests/test_migration.py` for validation
- **Type hints**: Added to key functions (partial)
- **Better error handling**: Improved error messages

#### Changed
- **requirements.txt**: Updated all dependencies
  - `torch>=2.2.0` (was 1.0.0)
  - `opencv-python>=4.8.0` (was 4.5.1.48)
  - `numpy>=1.24.0` (was unspecified)
  - Removed `tensorboardX`, `colorama`, `pytz`, `ffmpeg`
  - Added `imageio-ffmpeg`, official `tensorboard`
- **dataset.py**: Flow loading changed from `.flo` to `.npy` format
  - `LoadFlow()`: Now loads NumPy arrays
  - Removed `from flownet2 import flow_utils`
- **PyTorch API updates**:
  - Removed `Variable` wrapper (deprecated)
  - Replaced `upsample_bilinear` with `interpolate`
  - Updated `yaml.load` to `yaml.safe_load` (security fix)
- **README.md**: Complete rewrite of setup instructions
- **Device management**: Improved CUDA detection and CPU fallback

### Performance Improvements
- Flow generation: 2-3× faster
- Model loading: 20-30% faster with PyTorch 2.2
- Memory usage: -1GB VRAM (removed FlowNet2 overhead)

### Bug Fixes
- Fixed YAML security vulnerability (unsafe load)
- Fixed deprecation warnings in PyTorch 2.x
- Improved error messages for missing data

### Migration Guide
See `MIGRATION_REQUIREMENTS.md` for detailed migration instructions.

### Compatibility Notes
- Old model checkpoints (`.checkpoint` files) are fully compatible
- Flow files must be regenerated (new format)
- Python 3.10+ required (3.6-3.9 no longer supported)
- CUDA 11.8 or 12.1 recommended

---

## [1.0.0] - 2022-01-XX - Initial Release (WACV 2022)

Initial public release of "Deep Online Fused Video Stabilization" paper.

### Features
- Deep learning-based video stabilization
- Fusion of gyroscope, OIS, and optical flow
- FlowNet2 integration for optical flow
- Training and inference pipelines
- Evaluation metrics
```

#### R6.3: Update Configuration File Comments

**Requirement**: Add comments to `conf/stabilzation.yaml`

**Specification**: Add inline comments explaining each parameter
```yaml
data:
  exp: 'stabilzation'                    # Experiment name (used for checkpoint/log naming)
  checkpoints_dir: './checkpoint'        # Where to save model checkpoints
  log: './log'                           # Where to save training logs
  data_dir: './video'                    # Input data directory (for inference)
  use_cuda: true                         # Use GPU if available
  batch_size: 16                         # Number of videos per training batch
  resize_ratio: 0.25                     # Flow resolution scale (1920x1080 → 480x270)
  number_real: 10                        # Context window: ±10 frames of gyro data
  number_virtual: 2                      # Number of previous virtual predictions to use
  time_train: 2000                       # Training clip length (milliseconds)
  sample_freq: 40                        # Sampling frequency (milliseconds, 25 Hz)
  channel_size: 1                        # Number of input channels (if CNN enabled)
  num_workers: 16                        # DataLoader worker processes

model:
  load_model: null                       # Path to checkpoint to resume from (null = train from scratch)
  cnn:
    activate_function: relu              # Activation function for CNN layers
    batch_norm: true                     # Use batch normalization in CNN
    gap: false                           # Global average pooling
    layers: null                         # CNN layers config (null = disabled)
  rnn:
    layers:                              # LSTM layers configuration
    - - 512                              # Layer 1: 512 hidden units
      - true                             # Use bias
    - - 512                              # Layer 2: 512 hidden units
      - true                             # Use bias
  fc:
    activate_function: relu              # Activation for fully connected layers
    batch_norm: false                    # Batch norm for FC layers
    layers:                              # FC layers configuration
    - - 256                              # Hidden layer: 256 units
      - true                             # Use bias
    - - 4                                # Output layer: 4 units (quaternion)
      - true                             # Use bias
    drop_out: 0                          # Dropout probability (0 = disabled)

train:
  optimizer: "adam"                      # Optimizer (adam or sgd)
  momentum: 0.9                          # SGD momentum (if using sgd)
  decay_epoch: null                      # Manual decay epochs (null = use lr_step)
  epoch: 400                             # Total training epochs
  snapshot: 2                            # Save checkpoint every N epochs
  init_lr: 0.0001                        # Initial learning rate
  lr_decay: 0.5                          # Learning rate decay factor
  lr_step: 200                           # Decay every N epochs
  seed: 1                                # Random seed for reproducibility
  weight_decay: 0.0001                   # L2 regularization weight
  clip_norm: False                       # Gradient clipping (False = disabled)
  init: "xavier_uniform"                 # Weight initialization method

loss:
  follow: 10                             # Follow loss weight (stay close to real motion)
  angle: 1                               # Angle loss weight (penalize large deviations)
  smooth: 10                             # C1 smoothness loss weight
  c2_smooth: 200                         # C2 smoothness loss weight (strongest)
  undefine: 2.0                          # Undefined region loss weight (prevent black borders)
  opt: 0.1                               # Optical flow consistency loss weight
  stay: 0                                # Stay loss weight (penalize any rotation, usually 0)
```

### R7: Code Quality Improvements (Secondary Priority)

#### R7.1: Add Type Hints

**Requirement**: Add type hints to main functions

**Priority**: Low (nice-to-have)

**Example Changes**:
```python
# inference.py
def run(model: Model, loader: DataLoader, cf: dict, USE_CUDA: bool = True) -> np.ndarray:
    """
    Run inference on a video.
    
    Args:
        model: Trained stabilization model
        loader: DataLoader containing video data
        cf: Configuration dictionary
        USE_CUDA: Whether to use CUDA
        
    Returns:
        virtual_queue: Array of predicted quaternions, shape [N, 5] (timestamp + quat)
    """
    # ... implementation
```

#### R7.2: Improve Error Handling

**Requirement**: Add try-except blocks and informative errors

**Examples**:
```python
# dataset.py
def LoadFlow(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Flow directory not found: {path}\n"
            f"Have you run generate_flow.py to create optical flow files?"
        )
    
    file_names = [f for f in os.listdir(path) if f.endswith('.npy')]
    if len(file_names) == 0:
        raise ValueError(
            f"No .npy flow files found in {path}\n"
            f"Expected format: 00000.npy, 00001.npy, ...\n"
            f"Run: python generate_flow.py --data_dir <your_data_dir>"
        )
    # ... rest of function
```

#### R7.3: Add Configuration Validation

**Requirement**: Validate configuration file on load

**Implementation**:
```python
# util.py
def validate_config(cf: dict) -> None:
    """Validate configuration dictionary."""
    required_keys = ['data', 'model', 'train', 'loss']
    for key in required_keys:
        if key not in cf:
            raise ValueError(f"Missing required config section: {key}")
    
    # Validate data config
    data_cf = cf['data']
    if 'data_dir' not in data_cf:
        raise ValueError("data.data_dir is required")
    
    if not os.path.exists(data_cf['data_dir']):
        raise FileNotFoundError(f"data_dir does not exist: {data_cf['data_dir']}")
    
    # Validate model config
    # ... more validation
    
    print("✅ Configuration validated")
```

---

## ACCEPTANCE CRITERIA

### Must Have (Minimum Viable Product)

- ✅ **AC1**: Installation works with `pip install -r requirements.txt` on Python 3.10
- ✅ **AC2**: Inference completes without errors on test video
- ✅ **AC3**: Output files are created with correct naming
- ✅ **AC4**: No FlowNet2 dependencies remain in code
- ✅ **AC5**: README.md updated with new instructions
- ✅ **AC6**: Old checkpoints load successfully

### Should Have (Complete Migration)

- ✅ **AC7**: Flow generation is faster or equal to FlowNet2
- ✅ **AC8**: Stabilization quality metrics within 5% of original
- ✅ **AC9**: All PyTorch deprecation warnings resolved
- ✅ **AC10**: CHANGELOG.md documents all changes
- ✅ **AC11**: Basic test suite passes

### Nice to Have (Polish)

- ⭐ **AC12**: Type hints added to main functions
- ⭐ **AC13**: Improved error messages
- ⭐ **AC14**: CPU fallback mode works
- ⭐ **AC15**: Performance benchmarks documented

---

## VALIDATION PROCEDURE

### Step 1: Environment Setup
```bash
# Create clean environment
conda create -n dvs_test python=3.10
conda activate dvs_test

# Install dependencies
cd dvs
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

**Expected**: No errors, PyTorch ≥ 2.2.0, OpenCV ≥ 4.8.0

### Step 2: Flow Generation
```bash
# Download RAFT checkpoint
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth

# Generate flow for test video
python generate_flow.py --data_dir ./video --model raft-things.pth
```

**Expected**: Flow files created in `./video/s_114_outdoor_running_trail_daytime/flo/*.npy`

### Step 3: Inference
```bash
python inference.py --config ./conf/stabilzation.yaml --dir_path ./video
```

**Expected**:
- No errors
- Output in `./test/stabilzation/s_114_outdoor_running_trail_daytime/`
- Files: `*_stab.mp4`, `*.txt`, `*.jpg`

### Step 4: Visual Validation
```bash
# Play output video
ffplay ./test/stabilzation/s_114_outdoor_running_trail_daytime/s_114_outdoor_running_trail_daytime_stab.mp4
```

**Expected**: Smooth, stabilized video with no artifacts

### Step 5: Metrics
```bash
python metrics.py
```

**Expected**:
- Stability: <5 degrees/sec
- Cropping: >85%
- Distortion: <10 pixels
- No errors

### Step 6: Training (Optional)
```bash
# Use small subset for testing
python train.py --config ./conf/stabilzation_train.yaml
```

**Expected**:
- Training starts without errors
- Loss decreases over epochs
- Checkpoint saved

---

## ROLLBACK PLAN

If migration fails or introduces regressions:

### Option 1: Keep Both Versions
```
dvs/
├── flownet2/        # Old implementation (archived)
├── raft/            # New implementation
└── requirements_old.txt  # For old version
```

### Option 2: Git Branch Strategy
```bash
git branch migration_raft
git checkout migration_raft
# ... make changes
git checkout main  # Can always revert
```

### Option 3: Docker Containers
- Old environment: `Dockerfile.flownet2`
- New environment: `Dockerfile.raft`
- Users can choose which to use

---

## TIMELINE ESTIMATE

### Phase 1: Basic Migration (1-2 days)
- [x] Set up Python 3.10 environment
- [ ] Install PyTorch 2.2
- [ ] Integrate RAFT
- [ ] Update dataset.py for .npy loading
- [ ] Remove FlowNet2 code
- [ ] Test inference on one video

### Phase 2: API Updates (1 day)
- [ ] Remove Variable wrapper
- [ ] Replace upsample_bilinear
- [ ] Update yaml.load
- [ ] Fix deprecation warnings

### Phase 3: Testing (1-2 days)
- [ ] Create test suite
- [ ] Run full inference validation
- [ ] Compare flow quality
- [ ] Check checkpoint compatibility

### Phase 4: Documentation (1 day)
- [ ] Update README.md
- [ ] Create CHANGELOG.md
- [ ] Add code comments
- [ ] Create migration guide

### Phase 5: Polish (1 day, optional)
- [ ] Add type hints
- [ ] Improve error messages
- [ ] Optimize performance
- [ ] Add CPU fallback

**Total Estimated Time**: 5-7 days for complete migration

---

## SUPPORT AND TROUBLESHOOTING

### Common Issues

**Issue 1: RAFT import fails**
```
Solution: Ensure RAFT repository is cloned and in Python path
git clone https://github.com/princeton-vl/RAFT.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/RAFT/core
```

**Issue 2: CUDA out of memory**
```
Solution: Reduce batch size or use smaller input resolution
Edit conf/stabilzation.yaml:
  batch_size: 8  # Reduce from 16
  resize_ratio: 0.2  # Reduce from 0.25
```

**Issue 3: Old checkpoint doesn't load**
```
Solution: Check PyTorch version and try CPU loading
checkpoint = torch.load(path, map_location='cpu')
```

**Issue 4: Flow generation is slow**
```
Solution: Ensure CUDA is enabled for RAFT
device = torch.device('cuda')  # Not 'cpu'
```

### Contact and Resources

- **Original Paper**: [WACV 2022 Proceedings](https://openaccess.thecvf.com/content/WACV2022/papers/Shi_Deep_Online_Fused_Video_Stabilization_WACV_2022_paper.pdf)
- **RAFT Paper**: [ECCV 2020 Proceedings](https://arxiv.org/abs/2003.12039)
- **RAFT Repository**: https://github.com/princeton-vl/RAFT
- **PyTorch Migration Guide**: https://pytorch.org/docs/stable/migration_guide.html

---

## END OF MIGRATION REQUIREMENTS

This document specifies all technical requirements for the migration. For project context, see `PROJECT_CONTEXT.md`. For task breakdown, see `TASK_CHECKLIST.md`.
