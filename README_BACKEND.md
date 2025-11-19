# Neural Trainer Pro - Python Backend Edition

Train any image classification model using Python (PyTorch) backend with a React frontend.

## Features

✅ **Universal Dataset Support** - Automatically detects dataset structure  
✅ **Python Backend** - Uses PyTorch for fast GPU training  
✅ **Real-time Progress** - WebSocket updates during training  
✅ **Auto Structure Detection** - Supports multiple dataset formats  
✅ **GPU Accelerated** - CUDA support for NVIDIA GPUs  

## Setup Instructions

### 1. Install Node.js Dependencies

```bash
# Install frontend dependencies
npm install

# Install server dependencies
cd server
npm install
```

### 2. Install Python Dependencies

```bash
cd server
pip install -r requirements.txt
```

### 3. Start the Application

**Terminal 1 - Start Node.js Server:**
```bash
cd server
npm start
```

**Terminal 2 - Start React Frontend:**
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## Supported Dataset Structures

### 1. Train/Test Split Structure
```
dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img1.jpg
│       └── img2.jpg
└── test/
    ├── class1/
    └── class2/
```

### 2. Class Folders (Auto-split)
```
dataset/
├── class1/
│   ├── img1.jpg
│   └── img2.jpg
└── class2/
    ├── img1.jpg
    └── img2.jpg
```

### 3. CIFAR-10 Binary Format
```
dataset/
├── data_batch_1.bin
├── data_batch_2.bin
├── data_batch_3.bin
├── data_batch_4.bin
├── data_batch_5.bin
└── test_batch.bin
```

## How to Use

1. **Upload Dataset**: Select your dataset folder/files
2. **Configure Training**: Set epochs, batch size, learning rate
3. **Start Training**: Python backend trains the model
4. **Monitor Progress**: Real-time updates via WebSocket
5. **Export Model**: Download trained PyTorch model

## Technical Stack

- **Frontend**: React 18, Tailwind CSS, SweetAlert2
- **Backend**: Node.js, Express, WebSocket
- **ML**: Python, PyTorch, torchvision
- **Communication**: REST API + WebSocket

## API Endpoints

- `POST /api/upload-dataset` - Upload dataset files
- `POST /api/train` - Start training
- `GET /api/dataset-info/:path` - Get dataset structure info

## WebSocket Events

- `status` - Training status updates
- `info` - Dataset information
- `device` - Training device (CPU/CUDA)
- `epoch` - Epoch completion with metrics
- `complete` - Training finished
- `error` - Error messages

## Requirements

- Node.js 16+
- Python 3.8+
- (Optional) CUDA-capable GPU for faster training

## License

MIT
