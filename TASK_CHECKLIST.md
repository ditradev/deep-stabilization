# Deep Video Stabilization - Migration Task Checklist

## DOCUMENT PURPOSE

This document provides a **step-by-step actionable checklist** for migrating the Deep Video Stabilization project. Each task is designed to be independent and verifiable. Use this as your execution roadmap.

**How to Use**:
- [ ] Copy this file to track your progress
- [ ] Complete tasks in order (dependencies noted)
- [ ] Run verification commands after each task
- [ ] Mark tasks as complete: `- [ ]` → `- [x]`

---

## TASK OVERVIEW

```
Phase 1: Environment Setup          (5 tasks)  ⏱️ 2 hours
Phase 2: RAFT Integration           (7 tasks)  ⏱️ 4 hours
Phase 3: Remove FlowNet2            (4 tasks)  ⏱️ 1 hour
Phase 4: Update PyTorch APIs        (8 tasks)  ⏱️ 4 hours
Phase 5: Testing                    (6 tasks)  ⏱️ 8 hours
Phase 6: Documentation              (5 tasks)  ⏱️ 4 hours
Phase 7: Polish (Optional)          (5 tasks)  ⏱️ 4 hours

Total: 40 tasks, estimated 27 hours
```

---

## PHASE 1: ENVIRONMENT SETUP

### Task 1.1: Create Python 3.10 Environment

**Status**: [ ]

**Objective**: Set up isolated Python environment

**Steps**:
```bash
# Option A: Using Conda (recommended)
conda create -n dvs_modern python=3.10
conda activate dvs_modern

# Option B: Using venv
python3.10 -m venv dvs_env
source dvs_env/bin/activate  # Linux/Mac
# OR
dvs_env\Scripts\activate  # Windows
```

**Verification**:
```bash
python --version  # Should show Python 3.10.x
which python      # Should point to virtual environment
```

**Success Criteria**: Python 3.10.x is active in isolated environment

---

### Task 1.2: Install PyTorch 2.2 with CUDA

**Status**: [ ]

**Objective**: Install modern PyTorch with GPU support

**Dependencies**: Task 1.1

**Steps**:
```bash
# For CUDA 11.8
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

**Verification**:
```python
import torch
print(f"PyTorch: {torch.__version__}")  # Should be 2.2.0
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be True
print(f"CUDA version: {torch.version.cuda}")  # Should be 11.8 or 12.1
print(f"GPU: {torch.cuda.get_device_name(0)}")  # Should show your GPU
```

**Success Criteria**: PyTorch 2.2.0 installed with CUDA support confirmed

---

### Task 1.3: Create Updated requirements.txt

**Status**: [ ]

**Objective**: Define all dependencies in requirements file

**Dependencies**: None

**Steps**:
1. Navigate to `dvs/` directory
2. Create new `requirements.txt` or backup old one:
   ```bash
   mv requirements.txt requirements_old.txt
   ```
3. Create new `requirements.txt` with this content:

```txt
# Core deep learning
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24.0,<2.0.0

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

# Visualization
matplotlib>=3.7.0
pillow>=10.0.0

# RAFT will be added separately in Task 2.1
```

**Verification**:
```bash
cat requirements.txt  # Check content
wc -l requirements.txt  # Should be ~20 lines
```

**Success Criteria**: New requirements.txt created with updated dependencies

---

### Task 1.4: Install Dependencies

**Status**: [ ]

**Objective**: Install all Python packages

**Dependencies**: Tasks 1.1, 1.2, 1.3

**Steps**:
```bash
cd dvs/
pip install -r requirements.txt
```

**Verification**:
```python
# Test all imports
import torch
import torchvision
import cv2
import imageio
import numpy as np
import scipy
import yaml
import tqdm
import matplotlib
from PIL import Image

print("✅ All core dependencies imported successfully")
```

**Success Criteria**: All imports work without errors

---

### Task 1.5: Verify Current Code Runs (Baseline)

**Status**: [ ]

**Objective**: Confirm we can import existing code before changes

**Dependencies**: Task 1.4

**Steps**:
```python
# Test basic imports from existing code
cd dvs/
python -c "from model import Model, Net, UNet; print('Model imports OK')"
python -c "from dataset import Dataset_Gyro; print('Dataset imports OK')"
python -c "from gyro import torch_norm_quat; print('Gyro imports OK')"
```

**Expected Issues**:
- May see warnings about deprecated APIs
- FlowNet2 import errors are OK at this stage

**Verification**:
```bash
python -c "import yaml; cf = yaml.safe_load(open('conf/stabilzation.yaml')); print('Config loaded:', cf['data']['exp'])"
```

**Success Criteria**: Core modules import (FlowNet2 errors are acceptable)

---

## PHASE 2: RAFT INTEGRATION

### Task 2.1: Clone and Setup RAFT

**Status**: [ ]

**Objective**: Download RAFT repository and checkpoint

**Dependencies**: Task 1.4

**Steps**:
```bash
# Navigate to project root
cd ..  # Go to deep-stabilization/

# Clone RAFT
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT

# Install RAFT dependencies
pip install -r requirements.txt

# Download pretrained checkpoint
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth -O raft-things.pth

# Alternative if wget not available:
# curl -L https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth -o raft-things.pth

# Create checkpoint directory
mkdir -p ../dvs/raft_checkpoint
mv raft-things.pth ../dvs/raft_checkpoint/
```

**Verification**:
```bash
ls -lh dvs/raft_checkpoint/raft-things.pth  # Should be ~15MB
python -c "import sys; sys.path.append('RAFT/core'); from raft import RAFT; print('RAFT import OK')"
```

**Success Criteria**: RAFT repository cloned and checkpoint downloaded

---

### Task 2.2: Create RAFT Wrapper Class

**Status**: [ ]

**Objective**: Create Python class to use RAFT for flow estimation

**Dependencies**: Task 2.1

**Steps**:
1. Create new file: `dvs/flow_estimator.py`
2. Add this content:

```python
"""
Optical flow estimation using RAFT.
Replaces FlowNet2 functionality.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import cv2

# Add RAFT to path
RAFT_PATH = Path(__file__).parent.parent / 'RAFT' / 'core'
sys.path.insert(0, str(RAFT_PATH))

from raft import RAFT
from utils.utils import InputPadder


class RAFTFlowEstimator:
    """Wrapper for RAFT optical flow model."""
    
    def __init__(self, model_path='raft_checkpoint/raft-things.pth', device='cuda'):
        """
        Initialize RAFT flow estimator.
        
        Args:
            model_path: Path to RAFT checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        
        # Create RAFT model
        class Args:
            small = False
            mixed_precision = False
            alternate_corr = False
        
        self.model = RAFT(Args())
        
        # Load checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RAFT checkpoint not found: {model_path}\n"
                f"Download from: https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth"
            )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ RAFT model loaded on {device}")
    
    @torch.no_grad()
    def estimate_flow(self, frame1, frame2):
        """
        Estimate optical flow between two frames.
        
        Args:
            frame1: [H, W, 3] numpy array (uint8) or torch tensor
            frame2: [H, W, 3] numpy array (uint8) or torch tensor
            
        Returns:
            flow: [H, W, 2] numpy array (float32)
        """
        # Convert numpy to torch if needed
        if isinstance(frame1, np.ndarray):
            frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float()
            frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float()
        
        # Add batch dimension
        frame1 = frame1.unsqueeze(0).to(self.device)
        frame2 = frame2.unsqueeze(0).to(self.device)
        
        # Pad to multiple of 8
        padder = InputPadder(frame1.shape)
        frame1_pad, frame2_pad = padder.pad(frame1, frame2)
        
        # Estimate flow
        _, flow_up = self.model(frame1_pad, frame2_pad, iters=20, test_mode=True)
        
        # Unpad
        flow_up = padder.unpad(flow_up)
        
        # Convert to numpy [H, W, 2]
        flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
        
        return flow_np.astype(np.float32)


# Test function
if __name__ == '__main__':
    # Test with dummy frames
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    estimator = RAFTFlowEstimator()
    flow = estimator.estimate_flow(frame1, frame2)
    
    print(f"Flow shape: {flow.shape}")
    print(f"Flow range: [{flow.min():.2f}, {flow.max():.2f}]")
    print("✅ RAFT flow estimation test passed")
```

**Verification**:
```bash
cd dvs/
python flow_estimator.py
# Should output: "✅ RAFT model loaded on cuda"
#                "Flow shape: (480, 640, 2)"
#                "✅ RAFT flow estimation test passed"
```

**Success Criteria**: flow_estimator.py created and tested successfully

---

### Task 2.3: Create Flow Generation Script

**Status**: [ ]

**Objective**: Create script to generate flow for entire dataset

**Dependencies**: Task 2.2

**Steps**:
1. Create file: `dvs/generate_flow.py`
2. Add this content:

```python
"""
Generate optical flow for video dataset using RAFT.
Replaces: flownet2/run.sh

Usage:
    python generate_flow.py --data_dir ./video
    python generate_flow.py --data_dir ./dataset_release
"""

import argparse
import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from flow_estimator import RAFTFlowEstimator


def load_frames_from_video(video_path, max_frames=None):
    """Load frames from video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames


def load_frames_from_folder(frame_dir):
    """Load frames from extracted frame images."""
    frame_files = sorted(Path(frame_dir).glob('*.png'))
    if len(frame_files) == 0:
        frame_files = sorted(Path(frame_dir).glob('*.jpg'))
    
    frames = []
    for fpath in frame_files:
        frame = cv2.imread(str(fpath))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    return frames


def generate_flows(frames, flow_estimator, output_dir_fwd, output_dir_bwd):
    """Generate forward and backward flows for all frame pairs."""
    output_dir_fwd.mkdir(parents=True, exist_ok=True)
    output_dir_bwd.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating flows for {len(frames)-1} frame pairs...")
    
    for i in tqdm(range(len(frames) - 1), desc="Flow generation"):
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
    print(f"\n{'='*60}")
    print(f"Processing: {video_dir.name}")
    print(f"{'='*60}")
    
    # Find video file
    video_files = list(video_dir.glob('*.mp4'))
    if len(video_files) == 0:
        print(f"⚠️  No video file found in {video_dir}")
        return
    
    video_path = video_files[0]
    print(f"Video: {video_path.name}")
    
    # Check if frames folder exists
    frame_dir = video_dir / 'frames'
    if frame_dir.exists():
        print("Loading from frames folder...")
        frames = load_frames_from_folder(frame_dir)
    else:
        print("Loading from video file...")
        frames = load_frames_from_video(video_path)
    
    print(f"Loaded {len(frames)} frames")
    
    # Generate flows
    output_fwd = video_dir / 'flo'
    output_bwd = video_dir / 'flo_back'
    
    # Check if already exists
    if output_fwd.exists() and len(list(output_fwd.glob('*.npy'))) == len(frames) - 1:
        print(f"✅ Flows already exist, skipping...")
        return
    
    generate_flows(frames, flow_estimator, output_fwd, output_bwd)
    
    print(f"✅ Saved {len(frames)-1} flow pairs")
    print(f"   Forward:  {output_fwd}")
    print(f"   Backward: {output_bwd}")


def main():
    parser = argparse.ArgumentParser(description='Generate optical flow using RAFT')
    parser.add_argument('--data_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--model', default='raft_checkpoint/raft-things.pth', 
                       help='Path to RAFT checkpoint')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run on')
    args = parser.parse_args()
    
    # Initialize flow estimator
    print("="*60)
    print("Initializing RAFT flow estimator...")
    print("="*60)
    flow_estimator = RAFTFlowEstimator(model_path=args.model, device=args.device)
    
    # Find all video folders
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check for training/test split
    if (data_dir / 'training').exists() and (data_dir / 'test').exists():
        video_dirs = []
        video_dirs.extend(list((data_dir / 'training').iterdir()))
        video_dirs.extend(list((data_dir / 'test').iterdir()))
        video_dirs = [d for d in video_dirs if d.is_dir()]
    else:
        video_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(video_dirs)} video folders")
    
    # Process each video
    for video_dir in video_dirs:
        try:
            process_video_folder(video_dir, flow_estimator)
        except Exception as e:
            print(f"❌ Error processing {video_dir}: {e}")
            continue
    
    print("\n" + "="*60)
    print("✅ Flow generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
```

**Verification**:
```bash
python generate_flow.py --help
# Should show usage information
```

**Success Criteria**: generate_flow.py created with proper argument parsing

---

### Task 2.4: Test Flow Generation on Sample Video

**Status**: [ ]

**Objective**: Verify flow generation works on test data

**Dependencies**: Tasks 2.3

**Steps**:
```bash
cd dvs/

# Generate flow for sample video
python generate_flow.py --data_dir ./video

# This should process: video/s_114_outdoor_running_trail_daytime/
```

**Verification**:
```bash
# Check flow files were created
ls -lh video/s_114_outdoor_running_trail_daytime/flo/ | head
ls -lh video/s_114_outdoor_running_trail_daytime/flo_back/ | head

# Should see: 00000.npy, 00001.npy, ..., 00059.npy (or similar)

# Check file format
python -c "
import numpy as np
flow = np.load('video/s_114_outdoor_running_trail_daytime/flo/00000.npy')
print(f'Flow shape: {flow.shape}')
print(f'Flow dtype: {flow.dtype}')
print(f'Flow range: [{flow.min():.2f}, {flow.max():.2f}]')
"
# Expected output:
#   Flow shape: (270, 480, 2) or similar
#   Flow dtype: float32
#   Flow range: [-50.00, 50.00] or similar
```

**Success Criteria**: Flow .npy files generated successfully with correct shape

---

### Task 2.5: Update dataset.py - Remove FlowNet2 Import

**Status**: [ ]

**Objective**: Remove FlowNet2 dependency from dataset loading

**Dependencies**: Task 2.4

**Steps**:
1. Open `dvs/dataset.py`
2. **Line 22** - Delete this line:
   ```python
   from flownet2 import flow_utils
   ```

**Verification**:
```bash
grep "flownet2" dataset.py
# Should return empty (no matches)
```

**Success Criteria**: FlowNet2 import removed from dataset.py

---

### Task 2.6: Update dataset.py - Change LoadFlow Function

**Status**: [ ]

**Objective**: Update flow loading to use .npy format

**Dependencies**: Task 2.5

**Steps**:
1. Open `dvs/dataset.py`
2. Find `LoadFlow` function (around line 286)
3. Replace the entire function with:

```python
def LoadFlow(path):
    """
    Load optical flow file paths and shape.
    
    Args:
        path: Directory containing .npy flow files
    
    Returns:
        file_path: List of paths to flow files
        shape: (H, W) tuple of flow dimensions
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Flow directory not found: {path}\n"
            f"Run: python generate_flow.py --data_dir <your_data_dir>"
        )
    
    file_names = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
    file_path = [os.path.join(path, n) for n in file_names]
    
    if len(file_path) == 0:
        raise ValueError(
            f"No .npy files found in {path}\n"
            f"Expected format: 00000.npy, 00001.npy, ...\n"
            f"Run: python generate_flow.py --data_dir <your_data_dir>"
        )
    
    # Load first flow to get shape
    first_flow = np.load(file_path[0])
    shape = first_flow.shape[:2]  # (H, W)
    
    return file_path, shape
```

**Verification**:
```python
# Test the function
python -c "
import sys
sys.path.append('dvs')
from dataset import LoadFlow

paths, shape = LoadFlow('video/s_114_outdoor_running_trail_daytime/flo')
print(f'Loaded {len(paths)} flow files')
print(f'Flow shape: {shape}')
"
# Expected: "Loaded 60 flow files" (or similar)
#           "Flow shape: (270, 480)"
```

**Success Criteria**: LoadFlow function updated and tested

---

### Task 2.7: Update dataset.py - Change Flow Reading

**Status**: [ ]

**Objective**: Update flow reading in load_flo method

**Dependencies**: Task 2.6

**Steps**:
1. Open `dvs/dataset.py`
2. Find `load_flo` method (around line 179)
3. Find these lines (around 187-191):
   ```python
   for i in range(self.number_train):
       frame_id = i + first_id
       f = flow_utils.readFlow(self.data[idx].flo_path[frame_id-1]).astype(np.float32) 
       flo[i] = f

       f_b = flow_utils.readFlow(self.data[idx].flo_back_path[frame_id-1]).astype(np.float32) 
       flo_back[i] = f_b
   ```
4. Replace with:
   ```python
   for i in range(self.number_train):
       frame_id = i + first_id
       f = np.load(self.data[idx].flo_path[frame_id-1]).astype(np.float32)
       flo[i] = f

       f_b = np.load(self.data[idx].flo_back_path[frame_id-1]).astype(np.float32)
       flo_back[i] = f_b
   ```

**Verification**:
```bash
# Check syntax
python -c "import sys; sys.path.append('dvs'); from dataset import Dataset_Gyro; print('OK')"
```

**Success Criteria**: Flow reading updated to use np.load()

---

## PHASE 3: REMOVE FLOWNET2

### Task 3.1: Delete FlowNet2 Directory

**Status**: [ ]

**Objective**: Remove entire FlowNet2 codebase

**Dependencies**: Tasks 2.5, 2.6, 2.7 (ensure RAFT is working first)

**Steps**:
```bash
cd dvs/

# Backup first (optional but recommended)
tar -czf flownet2_backup.tar.gz flownet2/

# Remove directory
rm -rf flownet2/
```

**Verification**:
```bash
ls -d flownet2/
# Should output: "ls: flownet2/: No such file or directory"

# Verify no imports remain
grep -r "flownet2" --include="*.py" .
# Should return empty or only comments
```

**Success Criteria**: flownet2/ directory deleted

---

### Task 3.2: Search and Remove FlowNet2 References

**Status**: [ ]

**Objective**: Remove all textual references to FlowNet2

**Dependencies**: Task 3.1

**Steps**:
```bash
cd dvs/

# Search for any remaining references
grep -r "FlowNet2" --include="*.py" --include="*.md" --include="*.yaml" --include="*.sh" .

# Common locations to check manually:
# - README.md (sections about FlowNet2 installation)
# - Any .sh scripts
# - Comments in code
```

**Files to Update**:
1. `README.md` - Remove FlowNet2 Preparation section
2. Any shell scripts that reference FlowNet2

**Verification**:
```bash
grep -ri "flownet" . --include="*.py" --include="*.md"
# Should return minimal results (only historical references in docs)
```

**Success Criteria**: No functional FlowNet2 references remain

---

### Task 3.3: Update .gitignore

**Status**: [ ]

**Objective**: Update .gitignore for new flow format

**Dependencies**: None

**Steps**:
1. Open `.gitignore` (create if doesn't exist)
2. Add these lines:
   ```
   # Optical flow files
   **/flo/*.npy
   **/flo_back/*.npy
   
   # RAFT checkpoint
   dvs/raft_checkpoint/*.pth
   
   # Test outputs
   dvs/test/
   
   # Python
   __pycache__/
   *.pyc
   *.pyo
   
   # Environment
   dvs_env/
   .conda/
   ```

**Verification**:
```bash
cat .gitignore | grep "\.npy"
# Should show: **/flo/*.npy
```

**Success Criteria**: .gitignore updated

---

### Task 3.4: Clean Up Old .flo Files (if any)

**Status**: [ ]

**Objective**: Remove old .flo format files to avoid confusion

**Dependencies**: Task 2.4 (ensure .npy files exist first)

**Steps**:
```bash
# Find and list old .flo files
find dvs/video -name "*.flo" -type f

# If they exist and .npy files are confirmed working, remove them:
find dvs/video -name "*.flo" -type f -delete

# Or move to backup:
find dvs/video -name "*.flo" -type f -exec mv {} {}.old \;
```

**Verification**:
```bash
find dvs/video -name "*.flo" -type f
# Should return empty
```

**Success Criteria**: Old .flo files removed or backed up

---

## PHASE 4: UPDATE PYTORCH APIS

### Task 4.1: Remove Variable Import from model.py

**Status**: [ ]

**Objective**: Remove deprecated torch.autograd.Variable

**Dependencies**: None

**Steps**:
1. Open `dvs/model.py`
2. **Line 6** - Delete:
   ```python
   from torch.autograd import Variable
   ```

**Verification**:
```bash
grep "from torch.autograd import Variable" dvs/model.py
# Should return empty
```

**Success Criteria**: Variable import removed

---

### Task 4.2: Replace Variable() Calls in model.py

**Status**: [ ]

**Objective**: Remove Variable() wrapper from tensor creation

**Dependencies**: Task 4.1

**Steps**:
1. Open `dvs/model.py`
2. Search for `Variable(` - there should be 0 occurrences after this task
3. Replace each occurrence:

**Example replacements**:
```python
# Around line 37 in torch_norm_quat:
# OLD:
quat_out = Variable(torch.zeros((batch_size, 4), requires_grad=True))

# NEW:
quat_out = torch.zeros((batch_size, 4), requires_grad=True)

# The pattern is: just remove "Variable(" and the matching ")"
```

**Locations to update** (search each):
- `torch_norm_quat` function
- Any other functions creating Variables

**Verification**:
```bash
grep "Variable(" dvs/model.py
# Should return empty

python -c "import sys; sys.path.append('dvs'); from model import Model; print('OK')"
```

**Success Criteria**: All Variable() wrappers removed, code imports successfully

---

### Task 4.3: Replace Variable() in gyro/gyro_function.py

**Status**: [ ]

**Objective**: Remove Variable from quaternion operations

**Dependencies**: None

**Steps**:
1. Open `dvs/gyro/gyro_function.py`
2. **Line 6** - Remove import if present:
   ```python
   from torch.autograd import Variable
   ```
3. Search and replace all `Variable(` occurrences (approximately 7 locations):
   - `torch_norm_quat`
   - `torch_ConvertAxisAngleToQuaternion`
   - `torch_QuaternionProduct`
   - `torch_ConvertQuaternionToAxisAngle`
   - And other torch functions

**Pattern**:
```python
# OLD:
quat = Variable(torch.zeros((batch_size, 4), requires_grad=True))

# NEW:
quat = torch.zeros((batch_size, 4), requires_grad=True)
```

**Verification**:
```bash
grep "Variable" dvs/gyro/gyro_function.py
# Should return empty

python -c "import sys; sys.path.append('dvs'); from gyro import torch_norm_quat; print('OK')"
```

**Success Criteria**: All Variable uses removed from gyro functions

---

### Task 4.4: Replace upsample_bilinear in loss.py

**Status**: [ ]

**Objective**: Update to modern interpolate API

**Dependencies**: None

**Steps**:
1. Open `dvs/loss.py`
2. Find `torch.nn.functional.upsample_bilinear` (2 occurrences around lines 86, 89)
3. Replace:

```python
# OLD:
grid_t = torch.nn.functional.upsample_bilinear(grid_t, size=(h, w))

# NEW:
grid_t = torch.nn.functional.interpolate(grid_t, size=(h, w), mode='bilinear', align_corners=False)
```

**Verification**:
```bash
grep "upsample_bilinear" dvs/loss.py
# Should return empty

python -c "import sys; sys.path.append('dvs'); from loss import Optical_loss; print('OK')"
```

**Success Criteria**: upsample_bilinear replaced with interpolate

---

### Task 4.5: Update YAML Loading in train.py

**Status**: [ ]

**Objective**: Fix security vulnerability in YAML loading

**Dependencies**: None

**Steps**:
1. Open `dvs/train.py`
2. Find line ~153:
   ```python
   cf = yaml.load(open(config_file, 'r'))
   ```
3. Replace with:
   ```python
   with open(config_file, 'r') as f:
       cf = yaml.safe_load(f)
   ```

**Verification**:
```bash
grep "yaml.load" dvs/train.py
# Should return empty

grep "yaml.safe_load" dvs/train.py
# Should show the new line
```

**Success Criteria**: YAML loading uses safe_load

---

### Task 4.6: Update YAML Loading in inference.py

**Status**: [ ]

**Objective**: Fix security vulnerability in YAML loading

**Dependencies**: None

**Steps**:
1. Open `dvs/inference.py`
2. Find line ~190:
   ```python
   cf = yaml.load(open(config_file, 'r'))
   ```
3. Replace with:
   ```python
   with open(config_file, 'r') as f:
       cf = yaml.safe_load(f)
   ```

**Verification**:
```bash
grep "yaml.load" dvs/inference.py
# Should return empty
```

**Success Criteria**: YAML loading uses safe_load

---

### Task 4.7: Test All Imports After API Updates

**Status**: [ ]

**Objective**: Verify all modules import without deprecation warnings

**Dependencies**: Tasks 4.1-4.6

**Steps**:
```python
# Create test script: test_imports.py
import sys
import warnings
sys.path.append('dvs')

# Capture warnings
warnings.filterwarnings('error')

try:
    from model import Model, Net, UNet
    print("✅ model.py imports OK")
    
    from loss import C1_Smooth_loss, Optical_loss
    print("✅ loss.py imports OK")
    
    from dataset import Dataset_Gyro
    print("✅ dataset.py imports OK")
    
    from gyro import torch_norm_quat, torch_QuaternionProduct
    print("✅ gyro imports OK")
    
    import inference
    print("✅ inference.py imports OK")
    
    import train
    print("✅ train.py imports OK")
    
    print("\n🎉 All imports successful with no warnings!")
    
except Warning as w:
    print(f"⚠️  Warning detected: {w}")
except Exception as e:
    print(f"❌ Error: {e}")
```

Run:
```bash
python test_imports.py
```

**Success Criteria**: All imports work with no deprecation warnings

---

### Task 4.8: Run PyTorch Deprecation Check

**Status**: [ ]

**Objective**: Catch any remaining deprecated API uses

**Dependencies**: Task 4.7

**Steps**:
```bash
cd dvs/

# Run Python with warnings as errors
python -W error::DeprecationWarning -c "
import sys
sys.path.append('.')
from model import Model
from loss import Optical_loss
from dataset import Dataset_Gyro
print('✅ No deprecation warnings found')
"
```

**If warnings appear**: Go back and fix them

**Success Criteria**: No deprecation warnings detected

---

## PHASE 5: TESTING

### Task 5.1: Create Test Directory Structure

**Status**: [ ]

**Objective**: Set up testing infrastructure

**Dependencies**: None

**Steps**:
```bash
cd dvs/
mkdir -p tests
touch tests/__init__.py
```

**Verification**:
```bash
ls -la tests/
# Should show tests/ directory with __init__.py
```

**Success Criteria**: Test directory created

---

### Task 5.2: Create Unit Test Suite

**Status**: [ ]

**Objective**: Create automated tests for core functionality

**Dependencies**: Task 5.1

**Steps**:
1. Create `dvs/tests/test_migration.py`
2. Copy content from MIGRATION_REQUIREMENTS.md Section R5.1
3. The test should include:
   - test_imports()
   - test_model_creation()
   - test_quaternion_operations()
   - test_flow_loading()
   - test_model_forward()

**Verification**:
```bash
cd dvs/
python tests/test_migration.py
```

**Expected Output**:
```
✅ All imports successful
✅ Model created with 6847236 parameters
✅ Quaternion operations correct
✅ Flow loading works
✅ Model forward pass successful

🎉 All tests passed!
```

**Success Criteria**: All unit tests pass

---

### Task 5.3: Test Checkpoint Loading

**Status**: [ ]

**Objective**: Verify old checkpoints work in new environment

**Dependencies**: Task 5.2

**Steps**:
```python
# Create test script: test_checkpoint.py
import sys
sys.path.append('dvs')
import torch
import yaml

# Load config
with open('dvs/conf/stabilzation.yaml', 'r') as f:
    cf = yaml.safe_load(f)

# Load model
from model import Model
model = Model(cf)

# Load checkpoint
checkpoint_path = 'dvs/checkpoint/stabilzation/stabilzation_last.checkpoint'
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.net.load_state_dict(checkpoint['state_dict'])
    model.unet.load_state_dict(checkpoint['unet'])
    print("✅ Checkpoint loaded successfully")
    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
except FileNotFoundError:
    print("⚠️  Checkpoint not found - this is OK if you don't have pretrained model")
except Exception as e:
    print(f"❌ Checkpoint loading failed: {e}")
    raise

# Test inference
device = torch.device('cpu')  # Use CPU for testing
model.net.to(device)
model.unet.to(device)
model.net.eval()

# Dummy forward pass
batch_size = 1
model.net.init_hidden(batch_size)
input_dim = (2*10 + 1 + 2) * 4
inputs = torch.randn(batch_size, input_dim)
flo = torch.randn(batch_size, 270, 480, 2)
flo_back = torch.randn(batch_size, 270, 480, 2)
flo_out = model.unet(flo, flo_back)
ois = torch.randn(batch_size, 2)

with torch.no_grad():
    output = model.net(inputs, flo_out, ois)

print(f"✅ Model inference successful")
print(f"   Output shape: {output.shape}")
print(f"   Output quaternion: {output[0].tolist()}")
```

Run:
```bash
python test_checkpoint.py
```

**Success Criteria**: Checkpoint loads and inference runs

---

### Task 5.4: Run Full Inference Test

**Status**: [ ]

**Objective**: Test complete inference pipeline end-to-end

**Dependencies**: Tasks 2.4, 4.7, 5.3

**Steps**:
```bash
cd dvs/

# Ensure flow files exist
ls video/s_114_outdoor_running_trail_daytime/flo/*.npy | head -5

# Run inference
python inference.py --config ./conf/stabilzation.yaml --dir_path ./video
```

**Expected Behavior**:
- Should print "------Load Pretrained Model--------"
- Should print "Fininsh Load data" for each frame
- Should print "Step: 100/XXX" progress updates
- Should print loss values
- Should print "------Start Warping Video--------"
- Should complete without errors

**Verification**:
```bash
# Check output files were created
ls -lh test/stabilzation/s_114_outdoor_running_trail_daytime/

# Should see:
# - s_114_outdoor_running_trail_daytime.txt (trajectory)
# - s_114_outdoor_running_trail_daytime.jpg (plot)
# - s_114_outdoor_running_trail_daytime_stab.mp4 (video)

# Verify video plays
ffplay test/stabilzation/s_114_outdoor_running_trail_daytime/s_114_outdoor_running_trail_daytime_stab.mp4
```

**Success Criteria**: Full inference completes, output files created, video plays

---

### Task 5.5: Compare Output Quality

**Status**: [ ]

**Objective**: Verify stabilization quality is acceptable

**Dependencies**: Task 5.4

**Steps**:
1. **Visual Inspection**:
   - Play original: `video/s_114_outdoor_running_trail_daytime/*.mp4`
   - Play stabilized: `test/stabilzation/s_114_outdoor_running_trail_daytime/*_stab.mp4`
   - Check for:
     - Smooth camera motion (no jitter)
     - No excessive warping artifacts
     - No black borders (or reasonable borders)

2. **Trajectory Plot Inspection**:
   ```bash
   # View trajectory plot
   open test/stabilzation/s_114_outdoor_running_trail_daytime/s_114_outdoor_running_trail_daytime.jpg
   # OR on Linux: xdg-open ...
   ```
   - Blue line (stabilized) should be smoother than green (original)

3. **Metric Comparison** (if old results available):
   ```bash
   python metrics.py
   ```
   - Compare printed metrics with paper or old results

**Acceptance Criteria**:
- Video is visibly stabilized
- No major artifacts
- Trajectory is smoother than input

**Success Criteria**: Output quality is acceptable

---

### Task 5.6: Performance Benchmark

**Status**: [ ]

**Objective**: Measure and document performance improvements

**Dependencies**: Task 5.4

**Steps**:
```bash
# Time flow generation
time python generate_flow.py --data_dir ./video

# Time inference
time python inference.py --config ./conf/stabilzation.yaml --dir_path ./video

# Document results
```

Create `performance_results.txt`:
```
Hardware: [Your GPU, e.g., RTX 3090]
Video: s_114_outdoor_running_trail_daytime (60 frames, 1080p)

Flow Generation:
  - Time: X seconds
  - Speed: Y fps

Inference:
  - Time: X seconds  
  - Speed: Y fps

Total Time: X seconds

Comparison to Original:
  - FlowNet2: [if known]
  - RAFT: [measured above]
  - Speedup: [calculated]
```

**Success Criteria**: Performance metrics documented

---

## PHASE 6: DOCUMENTATION

### Task 6.1: Update README.md - Environment Section

**Status**: [ ]

**Objective**: Rewrite setup instructions

**Dependencies**: Task 5.4 (verify instructions work)

**Steps**:
1. Open `dvs/README.md`
2. Replace "Environment Setting" section (lines ~7-14) with:

```markdown
## Environment Setup

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support (11.8 or 12.1 recommended)
- 16GB+ GPU VRAM (for training), 8GB+ (for inference)

### Installation

```bash
# Create virtual environment (recommended)
conda create -n dvs python=3.10
conda activate dvs

# Install dependencies
cd dvs
pip install -r requirements.txt

# Clone and setup RAFT
cd ..
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
pip install -r requirements.txt

# Download RAFT checkpoint
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth
mv raft-things.pth ../dvs/raft_checkpoint/
cd ../dvs
```

**Installation complete!** No CUDA compilation required.
```

**Verification**:
```bash
grep "No CUDA compilation required" README.md
# Should find the line
```

**Success Criteria**: Installation section updated

---

### Task 6.2: Update README.md - Data Preparation

**Status**: [ ]

**Objective**: Update flow generation instructions

**Dependencies**: None

**Steps**:
1. Open `dvs/README.md`
2. Find "Data Preparation" section (lines ~16-24)
3. Replace with:

```markdown
## Data Preparation

Download sample video [here](https://drive.google.com/file/d/1PpF3-6BbQKy9fldjIfwa5AlbtQflx3sG/view?usp=sharing).
Extract the `video` folder under the `dvs` folder.

Generate optical flow:
```bash
cd dvs

# Generate optical flow using RAFT
python generate_flow.py --data_dir ./video --model raft_checkpoint/raft-things.pth
```

This will create `flo/` and `flo_back/` directories with `.npy` flow files.

Demo of curve visualization:
The **gyro/OIS curve visualization** can be found at `dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820_real.jpg`.
```

**Success Criteria**: Data preparation updated

---

### Task 6.3: Update README.md - Remove FlowNet2 Section

**Status**: [ ]

**Objective**: Remove obsolete instructions

**Dependencies**: None

**Steps**:
1. Open `dvs/README.md`
2. Find "FlowNet2 Preparation" section (lines ~26-37)
3. Replace entire section with:

```markdown
## ~~FlowNet2 Preparation~~

**OBSOLETE:** This section is no longer needed. The project now uses RAFT for optical flow, which requires no custom CUDA compilation.

For the old FlowNet2-based version, see git tag `v1.0-flownet2`.
```

**Success Criteria**: FlowNet2 section marked obsolete

---

### Task 6.4: Add Migration Notes to README

**Status**: [ ]

**Objective**: Document the migration for users

**Dependencies**: None

**Steps**:
1. Open `dvs/README.md`
2. Add new section before "Citation":

```markdown
## Migration Notes (v2.0)

This codebase has been modernized (March 2024):

### What Changed
- **Optical Flow**: FlowNet2 → RAFT
  - No more CUDA compilation required
  - 15MB model (vs 650MB FlowNet2)
  - 2-3× faster flow generation
- **Environment**: Python 3.6 + PyTorch 1.0 → Python 3.10 + PyTorch 2.2
- **Installation**: Single `pip install` command
- **Flow Format**: `.flo` (binary) → `.npy` (NumPy)

### Compatibility
- ✅ Old model checkpoints work without changes
- ⚠️ Flow files must be regenerated (new format)
- ⚠️ Python 3.10+ required (3.6-3.9 not supported)

### Performance
- Flow generation: 2-3× faster
- Training/inference: Same speed
- Memory: -1GB VRAM usage

For detailed migration guide, see `MIGRATION_REQUIREMENTS.md`.

### Accessing Old Version
The original FlowNet2-based code is available at git tag `v1.0-flownet2`.
```

**Success Criteria**: Migration notes added to README

---

### Task 6.5: Create CHANGELOG.md

**Status**: [ ]

**Objective**: Document all changes in structured format

**Dependencies**: None

**Steps**:
1. Create file: `CHANGELOG.md` in project root
2. Copy content from MIGRATION_REQUIREMENTS.md Section R6.2
3. Update date to actual migration completion date
4. Add any additional changes discovered during migration

**Verification**:
```bash
cat CHANGELOG.md | head -20
# Should show version 2.0.0 with migration changes
```

**Success Criteria**: CHANGELOG.md created with complete change log

---

## PHASE 7: POLISH (OPTIONAL)

### Task 7.1: Add Type Hints to Core Functions

**Status**: [ ]

**Objective**: Improve code documentation and IDE support

**Dependencies**: None (optional task)

**Priority**: Low

**Steps**:
1. Add type hints to main functions in:
   - `inference.py::run()`,
   - `train.py::run_epoch()`
   - `model.py::forward()` methods
   - `dataset.py::__getitem__()`

**Example**:
```python
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

def run(
    model: Model, 
    loader: DataLoader, 
    cf: Dict, 
    USE_CUDA: bool = True
) -> np.ndarray:
    """Run inference on video."""
    # ... implementation
```

**Success Criteria**: Type hints added to at least 10 main functions

---

### Task 7.2: Improve Error Messages

**Status**: [ ]

**Objective**: Add helpful error messages for common issues

**Dependencies**: None (optional task)

**Priority**: Low

**Steps**:
Add try-except blocks with informative messages:

```python
# dataset.py
def process_one_video(self, path):
    dvs_data = DVS_data()
    files = sorted(os.listdir(path))
    
    # Check for required files
    if not any('gyro' in f for f in files):
        raise FileNotFoundError(
            f"No gyro file found in {path}\n"
            f"Expected file: *_gyro.txt\n"
            f"Available files: {files}"
        )
    
    # ... rest of function
```

**Success Criteria**: Improved error messages for at least 5 common failure points

---

### Task 7.3: Add Configuration Validation

**Status**: [ ]

**Objective**: Catch configuration errors early

**Dependencies**: None (optional task)

**Priority**: Low

**Steps**:
1. Create `dvs/util.py::validate_config()` function
2. Call from `train.py` and `inference.py` after loading config
3. Check for:
   - Required keys exist
   - Paths are valid
   - Numerical values in valid ranges

**Example**:
```python
def validate_config(cf: dict) -> None:
    """Validate configuration dictionary."""
    # Check required sections
    required = ['data', 'model', 'train', 'loss']
    for key in required:
        if key not in cf:
            raise ValueError(f"Config missing required section: '{key}'")
    
    # Check data directory exists
    if not os.path.exists(cf['data']['data_dir']):
        raise FileNotFoundError(
            f"data_dir not found: {cf['data']['data_dir']}\n"
            f"Please set correct path in config file"
        )
    
    # Check numerical ranges
    if cf['data']['batch_size'] < 1:
        raise ValueError("batch_size must be >= 1")
    
    print("✅ Configuration validated")
```

**Success Criteria**: Configuration validation implemented

---

### Task 7.4: Add CPU Fallback Mode

**Status**: [ ]

**Objective**: Allow running on CPU (slow but possible)

**Dependencies**: None (optional task)

**Priority**: Low

**Steps**:
1. Create device management helper:

```python
# util.py
def get_device(config: dict) -> torch.device:
    """Get device based on config and availability."""
    use_cuda = config['data'].get('use_cuda', True)
    
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA requested but not available, falling back to CPU")
            print("   Warning: CPU inference is 10-50× slower")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using CPU (as configured)")
    
    return device
```

2. Replace hardcoded `.cuda()` calls with `.to(device)`:
   - This is a large refactor (100+ occurrences)
   - Consider using regex: find `\.cuda\(\)`, replace with `.to(device)`
   - Must pass `device` variable to all relevant functions

**Success Criteria**: Code runs on CPU (even if slowly)

---

### Task 7.5: Add Progress Bars to Flow Generation

**Status**: [ ]

**Objective**: Better user experience during flow generation

**Dependencies**: None (optional task)

**Priority**: Low

**Steps**:
Already implemented in Task 2.3 (generate_flow.py uses tqdm)

Verify:
```bash
python generate_flow.py --data_dir ./video
# Should show progress bar: "Flow generation: 45%|████▌     | 27/60 [00:12<00:15,  2.1it/s]"
```

**Success Criteria**: Progress bars work correctly

---

## FINAL VALIDATION

### Final Check 1: Clean Install Test

**Status**: [ ]

**Objective**: Verify installation from scratch

**Steps**:
```bash
# Create completely new environment
conda create -n dvs_final_test python=3.10 -y
conda activate dvs_final_test

# Follow README.md instructions exactly
cd dvs
pip install -r requirements.txt
cd ..
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT && pip install -r requirements.txt
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth
mv raft-things.pth ../dvs/raft_checkpoint/

# Test inference
cd ../dvs
python generate_flow.py --data_dir ./video
python inference.py --config ./conf/stabilzation.yaml --dir_path ./video
```

**Success Criteria**: Works perfectly following only README.md

---

### Final Check 2: Code Quality Scan

**Status**: [ ]

**Objective**: Check for any remaining issues

**Steps**:
```bash
cd dvs/

# Check for FlowNet2 references
grep -ri "flownet" --include="*.py" --include="*.md"
# Should return minimal/no results

# Check for Variable usage
grep -r "from torch.autograd import Variable" --include="*.py"
# Should return empty

# Check for old flow format
grep -r "flow_utils.readFlow" --include="*.py"
# Should return empty

# Check for deprecated APIs
grep -r "upsample_bilinear" --include="*.py"
# Should return empty
```

**Success Criteria**: No problematic patterns found

---

### Final Check 3: Documentation Complete

**Status**: [ ]

**Objective**: Verify all documentation is updated

**Checklist**:
- [ ] README.md updated with new instructions
- [ ] README.md has migration notes
- [ ] CHANGELOG.md exists and complete
- [ ] PROJECT_CONTEXT.md present (reference)
- [ ] MIGRATION_REQUIREMENTS.md present (reference)
- [ ] requirements.txt updated
- [ ] .gitignore includes .npy files

**Success Criteria**: All documentation complete

---

### Final Check 4: Git Commit and Tag

**Status**: [ ]

**Objective**: Create clean git history

**Steps**:
```bash
# Review changes
git status
git diff

# Stage changes
git add dvs/requirements.txt
git add dvs/flow_estimator.py
git add dvs/generate_flow.py
git add dvs/model.py
git add dvs/dataset.py
git add dvs/inference.py
git add dvs/train.py
git add dvs/loss.py
git add dvs/gyro/gyro_function.py
git add dvs/README.md
git add CHANGELOG.md
git add .gitignore

# Remove deleted files
git rm -r dvs/flownet2

# Commit
git commit -m "Migration to Python 3.10, PyTorch 2.2, and RAFT

Major changes:
- Replaced FlowNet2 with RAFT for optical flow
- Upgraded to Python 3.10 and PyTorch 2.2
- Removed all custom CUDA compilation requirements
- Updated deprecated PyTorch APIs
- Simplified installation to single pip command
- Changed flow format from .flo to .npy

See CHANGELOG.md for detailed changes.
"

# Tag release
git tag -a v2.0.0 -m "Version 2.0.0: Modern stack migration"
```

**Success Criteria**: Clean commit created with tag

---

## COMPLETION CHECKLIST

Mark complete when ALL tasks in a phase are done:

- [ ] **Phase 1: Environment Setup** (5/5 tasks)
- [ ] **Phase 2: RAFT Integration** (7/7 tasks)
- [ ] **Phase 3: Remove FlowNet2** (4/4 tasks)
- [ ] **Phase 4: Update PyTorch APIs** (8/8 tasks)
- [ ] **Phase 5: Testing** (6/6 tasks)
- [ ] **Phase 6: Documentation** (5/5 tasks)
- [ ] **Phase 7: Polish** (5/5 tasks) - OPTIONAL

**Required for MVP**: Phases 1-6 complete
**Complete Migration**: Phases 1-7 complete

---

## SUCCESS CRITERIA SUMMARY

### Minimum Viable Product (Must Have)
- ✅ Python 3.10 + PyTorch 2.2 environment works
- ✅ RAFT optical flow generation works
- ✅ FlowNet2 completely removed
- ✅ Inference runs without errors
- ✅ Output videos are generated
- ✅ README.md updated

### Complete Migration (Should Have)
- ✅ All deprecated APIs updated
- ✅ Unit tests pass
- ✅ Performance documented
- ✅ CHANGELOG.md created
- ✅ Old checkpoints work

### Polished Release (Nice to Have)
- ⭐ Type hints added
- ⭐ Better error messages
- ⭐ CPU fallback works
- ⭐ Code quality scan passes

---

## TROUBLESHOOTING GUIDE

### Issue: RAFT import fails
```
Error: ModuleNotFoundError: No module named 'raft'
```
**Solution**:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/RAFT/core
# Or add to flow_estimator.py:
sys.path.insert(0, '../RAFT/core')
```

### Issue: CUDA out of memory
```
Error: RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in config:
```yaml
data:
  batch_size: 4  # Reduce from 16
  resize_ratio: 0.2  # Reduce from 0.25
```

### Issue: Flow shape mismatch
```
Error: RuntimeError: shape mismatch
```
**Solution**: Regenerate flow files:
```bash
rm -rf video/*/flo video/*/flo_back
python generate_flow.py --data_dir ./video
```

### Issue: Checkpoint won't load
```
Error: KeyError or size mismatch
```
**Solution**: 
```python
checkpoint = torch.load(path, map_location='cpu')
# Check if keys match:
print(checkpoint.keys())
# Try strict=False:
model.load_state_dict(checkpoint['state_dict'], strict=False)
```

---

## END OF TASK CHECKLIST

**How to Report Completion**:
1. Mark all tasks as [x] completed
2. Run Final Validation checks
3. Create git commit and tag
4. Document any deviations or issues encountered
5. Share performance benchmarks

**Estimated Total Time**: 25-35 hours for complete migration (including Phase 7)

**Questions or Issues**: Refer to:
- `PROJECT_CONTEXT.md` for code understanding
- `MIGRATION_REQUIREMENTS.md` for technical specifications
- Original paper for algorithm details

