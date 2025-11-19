# Quick Start Guide

## Complete System Rewrite ✅

Your application has been completely converted from **TensorFlow.js** (browser-based) to **Python backend** (PyTorch) training!

## What Changed

### ✅ Removed
- TensorFlow.js - No more browser training
- CIFAR-10 specific code - Now supports ANY dataset
- GPU browser issues - Python handles GPU properly

### ✅ Added
- **Node.js Express Server** (`server/server.js`)
- **Python Training Script** (`server/train.py`) with PyTorch
- **WebSocket** for real-time updates
- **Automatic dataset structure detection**
- **Universal dataset support**

## Files Created

```
server/
├── server.js          # Express API server
├── train.py           # Python PyTorch training
├── package.json       # Server dependencies
└── requirements.txt   # Python dependencies

src/
└── AppNew.jsx        # New React app with API calls
```

## Setup Steps

### 1. Run Setup Script
```powershell
.\setup.ps1
```

This installs:
- Frontend dependencies (React, axios)
- Server dependencies (Express, multer, ws)
- Python dependencies (torch, torchvision, Pillow)

### 2. Start Backend Server
```bash
cd server
npm start
```
Server runs on `http://localhost:3001`  
WebSocket runs on `ws://localhost:3002`

### 3. Start Frontend
```bash
npm run dev
```
Frontend runs on `http://localhost:5173`

## How It Works

1. **Upload Dataset** → Files sent to Node.js server
2. **Structure Detection** → Server analyzes folder structure
3. **Start Training** → Server spawns Python process
4. **Python Trains** → PyTorch trains on CPU/GPU
5. **Real-time Updates** → WebSocket sends epoch data to frontend
6. **Training Complete** → Model saved as `.pth` file

## Supported Dataset Formats

### Format 1: Train/Test Folders
```
my_dataset/
├── train/
│   ├── cats/
│   ├── dogs/
│   └── birds/
└── test/
    ├── cats/
    ├── dogs/
    └── birds/
```

### Format 2: Class Folders (Auto-split 80/20)
```
my_dataset/
├── cats/
├── dogs/
└── birds/
```

### Format 3: CIFAR-10 Binary
```
cifar-10/
├── data_batch_1.bin
├── data_batch_2.bin
...
```

## Key Features

✅ **Any Dataset** - Upload any image classification dataset  
✅ **Auto Detection** - Automatically identifies structure  
✅ **GPU Training** - Uses CUDA if available  
✅ **Real-time Updates** - See training progress live  
✅ **PyTorch Models** - Professional ML framework  
✅ **No Browser Limits** - Python handles everything  

## Next Steps

1. Replace `src/App.jsx` with `src/AppNew.jsx`
2. Update `src/main.jsx` to remove TensorFlow.js initialization
3. Update components to work with new data structure
4. Test with your dataset!

## API Endpoints

- `POST /api/upload-dataset` - Upload files
- `POST /api/train` - Start training
- WebSocket: Real-time progress updates

## Need Help?

Check `README_BACKEND.md` for full documentation!
