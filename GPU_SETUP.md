# GPU Installation Guide

## ✅ Your System
**Detected GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
**Python Version:** 3.14.0
**Current PyTorch:** 2.9.1+cpu (CPU-only)

## ⚠️ Current Issue
PyTorch CUDA builds are **not yet available for Python 3.14** (too new!)
Your training script is already GPU-optimized, but using CPU until PyTorch CUDA support is available.

## 🔧 Solutions (Choose One)

### Option 1: Wait for PyTorch CUDA Support (Recommended for now)
- PyTorch team will release CUDA builds for Python 3.14 soon
- Your code is already GPU-ready, no changes needed
- Check back at: https://pytorch.org/get-started/locally/
- Monitor: https://github.com/pytorch/pytorch/issues

### Option 2: Downgrade to Python 3.11 (Works Now)
**Steps:**
1. Install Python 3.11 from python.org
2. Delete current virtual environment: `Remove-Item -Recurse .venv`
3. Create new venv: `python3.11 -m venv .venv`
4. Activate: `.venv\Scripts\activate`
5. Install dependencies: `pip install -r requirements.txt` (if you have one)
6. Install PyTorch with CUDA 11.8:
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
7. Verify: `python server/check_gpu.py`

### Option 3: Use Conda (Alternative)
```bash
# Install Miniconda/Anaconda
conda create -n ml_env python=3.11
conda activate ml_env
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

## ✅ GPU-Ready Features Already Implemented
Your training script (`train.py`) already has:
- ✅ Automatic GPU detection
- ✅ CUDA device selection
- ✅ Pin memory for faster data transfer
- ✅ Non-blocking GPU operations
- ✅ Parallel data loading (num_workers=2)
- ✅ GPU memory cleanup after training
- ✅ Real-time GPU info display

## 📊 Expected Performance (Once GPU is enabled)
- **Training speed**: 10-50x faster
- **Batch processing**: 5-20x improvement
- **Your RTX 3050 specs**: Perfect for small-to-medium models
- **4GB VRAM**: Can handle batch sizes of 32-64 easily

## 🔍 Verify GPU After Setup
Run: `python server/check_gpu.py`

Should show:
```
✓ CUDA available: True
✓ GPU name: NVIDIA GeForce RTX 3050 Laptop GPU
✓ Total Memory: 4.00 GB
```

## 💡 Temporary Workaround
Your training will work fine on CPU (just slower). The code will automatically switch to GPU once PyTorch CUDA is installed - no code changes needed!
