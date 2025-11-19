# GPU Detection Issue - Solution

## Current Problem
You have an NVIDIA RTX 3050 GPU, but PyTorch can't detect it because:
- Python version: 3.14.0 (too new!)
- PyTorch version: 2.9.1+cpu (CPU-only)
- PyTorch CUDA builds not available for Python 3.14 yet

## Solution: Downgrade to Python 3.11

### Step 1: Install Python 3.11
1. Download Python 3.11 from: https://www.python.org/downloads/
2. Install it (check "Add to PATH")

### Step 2: Recreate Virtual Environment
```powershell
# Delete current virtual environment
Remove-Item -Recurse -Force .venv

# Create new venv with Python 3.11
python3.11 -m venv .venv

# Or if python3.11 doesn't work, use full path:
# C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv

# Activate the new environment
.venv\Scripts\activate

# Verify Python version
python --version  # Should show Python 3.11.x
```

### Step 3: Install PyTorch with CUDA
```powershell
# Install PyTorch with CUDA 11.8 (best compatibility)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or try CUDA 12.1 (latest)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Other Dependencies
```powershell
pip install pillow numpy
```

### Step 5: Verify GPU Detection
```powershell
python server/check_gpu.py
```

You should see:
```
✓ CUDA available: True
✓ GPU name: NVIDIA GeForce RTX 3050 Laptop GPU
✓ Total Memory: 4.00 GB
```

### Step 6: Install Model Export Dependencies (Optional)
```powershell
# For ONNX export
pip install onnx

# For TorchScript (built-in, no install needed)

# For CoreML (macOS only)
pip install coremltools

# For TensorFlow Lite
pip install onnx-tf tensorflow
```

## Expected Results
After completing these steps:
- ✅ GPU will be automatically detected
- ✅ Training will be 10-50x faster
- ✅ Device will show: "GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4.0GB)"
- ✅ Model export to all formats will work

## Current Workarounds
Until you downgrade Python:
- ✅ Training works on CPU (just slower)
- ✅ PyTorch format export works
- ❌ Other formats need dependencies installed (see error messages)
