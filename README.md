# 🧠 Neural Trainer Pro

Train any image classification model using Python + PyTorch with a beautiful React interface!

## ✨ Features

- 🚀 **Single Command Start** - `npm run dev` starts everything
- 🐍 **Python Backend** - Professional PyTorch training
- 🎨 **Clean UI** - Simple 3-step workflow
- 📊 **Real-time Progress** - WebSocket updates during training
- 🔄 **Auto Dataset Detection** - Works with any folder structure
- 💻 **GPU/CPU Auto** - Uses CUDA if available
- 🎯 **Universal** - Train any image classification model

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ installed
- Python 3.8+ installed
- pip (Python package manager)

### Installation

1. **Install Python Dependencies**
```bash
cd server
pip install torch torchvision Pillow numpy
cd ..
```

2. **Install Node.js Dependencies** (if not already done)
```bash
npm install
```

3. **Start Everything**
```bash
npm run dev
```

4. **Open Browser**
```
http://localhost:5174  (or whatever port Vite shows)
```

## 📖 How to Use

### Step 1: Upload Dataset
1. Click "Select Dataset Folder"
2. Choose your image classification dataset
3. System auto-detects structure (4 formats supported)
4. See dataset info (classes, images, split)

### Step 2: Configure Training
1. Set **Epochs** (how many times to train on full dataset)
2. Set **Batch Size** (images processed together)
3. Set **Learning Rate** (training speed)
4. Click "Start Training"

### Step 3: Watch Training
1. Real-time progress updates
2. See current epoch, loss, accuracy
3. GPU/CPU status shown
4. Model saves automatically when done

## 📁 Supported Dataset Formats

### Format 1: Simple Class Folders ✅ RECOMMENDED
```
my_dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── cat3.jpg
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── dog3.jpg
└── birds/
    ├── bird1.jpg
    └── bird2.jpg
```
**Perfect for:** Custom datasets, quick testing

### Format 2: Train/Test Split
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
**Perfect for:** Pre-split datasets, Kaggle competitions

### Format 3: CIFAR-10 Binary
```
cifar-10/
├── data_batch_1.bin
├── data_batch_2.bin
├── data_batch_3.bin
├── data_batch_4.bin
├── data_batch_5.bin
└── test_batch.bin
```
**Perfect for:** CIFAR-10 dataset, benchmarking

### Format 4: Flat Images
```
my_dataset/
├── image1.jpg
├── image2.png
└── image3.jpg
```
**Perfect for:** Single-class datasets, preprocessing testing

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │  (Port 5173/5174)
│   (Vite Dev)    │
└────────┬────────┘
         │ HTTP/WebSocket
         ▼
┌─────────────────┐
│  Node.js Server │  (Port 3001)
│    (Express)    │
└────────┬────────┘
         │ Spawn Process
         ▼
┌─────────────────┐
│ Python Training │
│    (PyTorch)    │
└─────────────────┘
```

### Tech Stack

**Frontend:**
- React 18.3.1
- Vite 5.4.8
- Axios 1.6.0
- SweetAlert2 11.26.3
- Tailwind CSS 3.4.14

**Backend:**
- Node.js + Express 4.18.2
- WebSocket (ws 8.14.2)
- Multer 1.4.5 (file uploads)

**ML Training:**
- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- Pillow (image processing)

## 📂 Project Structure

```
TrainModelUsingJs/
├── server/
│   ├── server.js           # Express API server
│   ├── train.py            # PyTorch training script
│   ├── package.json        # Server dependencies
│   ├── requirements.txt    # Python dependencies
│   └── uploads/            # Uploaded datasets (created auto)
├── src/
│   ├── App.jsx             # Main application component
│   ├── main.jsx            # Entry point
│   ├── components/
│   │   ├── DatasetSelector.jsx    # File upload UI
│   │   ├── TrainingConfig.jsx     # Training parameters
│   │   └── TrainingProgress.jsx   # Real-time progress
│   ├── styles/
│   │   └── index.css       # Tailwind styles
│   └── ...
├── package.json            # Root dependencies & scripts
├── vite.config.js          # Vite configuration
├── tailwind.config.js      # Tailwind configuration
└── README.md               # This file
```

## 🔧 Configuration

### Training Parameters

**Epochs** (default: 10)
- How many times the model sees the entire dataset
- More epochs = potentially better accuracy (but can overfit)
- Range: 1-100

**Batch Size** (default: 32)
- Number of images processed together
- Larger = faster but needs more memory
- Range: 8-128

**Learning Rate** (default: 0.001)
- How fast the model learns
- Too high = unstable training
- Too low = very slow learning
- Range: 0.0001-0.01

### Model Architecture

Default model (SimpleCNN):
```
Conv2D(32) → MaxPool → Dropout
Conv2D(64) → MaxPool → Dropout
Conv2D(128) → MaxPool → Dropout
Dense(128) → Dropout
Output (num_classes)
```

To customize, edit `server/train.py`

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill processes on ports
npx kill-port 3001 3002 5173 5174

# Then restart
npm run dev
```

### Python Not Found
```bash
# Check Python installation
python --version
# or
python3 --version

# Should show 3.8 or higher
```

### PyTorch Installation Issues
```bash
# For CPU-only (smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### WebSocket Connection Failed
- Check if server is running on port 3001
- Check console for errors
- Refresh browser page

### Training Starts But No Progress
- Check server terminal for Python errors
- Ensure dataset is valid (has images)
- Check Python dependencies are installed

## 📊 What Happens During Training?

1. **Upload Dataset**
   - Files uploaded to `server/uploads/`
   - Structure analyzed automatically
   - Classes and splits detected

2. **Start Training**
   - Node.js spawns Python process
   - PyTorch loads dataset
   - GPU/CPU detected and used
   - Model initialized

3. **Training Loop**
   - Each epoch:
     - Forward pass (predictions)
     - Loss calculation
     - Backward pass (gradients)
     - Weight updates
   - Progress sent via WebSocket

4. **Completion**
   - Model saved as `.pth` file
   - Final accuracy displayed
   - Ready for next training

## 🎯 Example Workflow

```bash
# 1. Prepare your dataset
# Put images in folders by class:
# my_animals/
#   ├── cats/
#   ├── dogs/
#   └── birds/

# 2. Start the app
npm run dev

# 3. In browser:
#    - Upload my_animals folder
#    - Set epochs to 20
#    - Set batch size to 32
#    - Click "Start Training"

# 4. Wait for training
#    - Watch real-time progress
#    - See accuracy improve

# 5. Model saved!
#    - Located in server/uploads/
#    - File: trained_model_TIMESTAMP.pth
```

## 🌟 Tips for Best Results

1. **More Data = Better Model**
   - At least 100 images per class recommended
   - More variety in images helps generalization

2. **Balance Your Classes**
   - Try to have similar number of images per class
   - Imbalanced data can bias the model

3. **Image Quality**
   - Use clear, well-lit images
   - Consistent image sizes help
   - Remove corrupted images

4. **Training Settings**
   - Start with 10 epochs
   - Increase if accuracy still improving
   - Lower learning rate if training unstable

5. **GPU vs CPU**
   - GPU is 10-100x faster
   - CPU works fine for small datasets
   - Auto-detected (no configuration needed)

## 📝 Credits

Built with ❤️ using:
- React & Vite
- PyTorch
- Node.js & Express
- TailwindCSS
- SweetAlert2

## 📄 License

MIT License - Feel free to use for any project!

---

**Need Help?** Check the troubleshooting section or open an issue!

**Want to Customize?** All code is modular and well-commented!

Happy Training! 🚀
