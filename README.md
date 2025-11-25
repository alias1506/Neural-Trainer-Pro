<div align="center">

# Neural Trainer Pro

### Enterprise-Grade Machine Learning Training Platform

*Train, visualize, and export deep learning models with professional-grade tools and real-time insights*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 16+](https://img.shields.io/badge/node.js-16+-green.svg)](https://nodejs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)

</div>

---

## 🚀 Overview

Neural Trainer Pro is a production-ready machine learning training platform that seamlessly integrates PyTorch's computational power with a modern React interface. Deploy custom image classification models with enterprise-level visualization, automated workflows, and multi-format export capabilities.

### ✨ What Makes It Special

- **Zero Configuration** - Upload datasets and start training immediately
- **Real-Time Insights** - Watch your model learn with live metrics and interactive charts
- **Universal Export** - Deploy to any platform with multiple export formats
- **Professional UI** - Clean, intuitive interface built with modern design principles
- **GPU Accelerated** - Automatic hardware detection for maximum performance

## Key Features

### 🎯 **Training & Visualization**
- Real-time performance metrics with interactive charts
- GPU acceleration with automatic hardware detection
- Comprehensive training history and session management
- Live progress monitoring with WebSocket integration
- Advanced console logging for detailed insights

### 📦 **Model Export Flexibility**
Export trained models to industry-standard formats:
- **PyTorch (.pth)** - Native deployment
- **ONNX (.onnx)** - Cross-platform inference
- **TorchScript (.pt)** - Production optimization
- **CoreML (.mlmodel)** - iOS/macOS integration *(Optimization in Progress)*
- **TensorFlow Lite (.tflite)** - Mobile & edge devices *(Optimization in Progress)*

> **Note:** TFLite and CoreML exports are currently marked as "Work in Progress" in the UI while we optimize dependency compatibility.

### 🛠️ **Professional Workflow**
- Intuitive drag-and-drop dataset upload
- Automatic dataset validation and analysis
- Configurable hyperparameters (epochs, batch size, learning rate)
- Intelligent resource cleanup and management
- Responsive, production-ready interface

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18, Vite, Chart.js, Tailwind CSS, SweetAlert2 |
| **Backend** | Node.js, Express, WebSocket |
| **ML Engine** | PyTorch, ONNX |
| **Infrastructure** | GPU/CUDA Support, Virtual Environments |

## 🎯 Quick Start

### Prerequisites
- **Node.js** 16+ ([Download](https://nodejs.org/))
- **Python** 3.11+ ([Download](https://www.python.org/))
- **CUDA GPU** (optional, for faster training)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/alias1506/Neural-Trainer-Pro.git
cd Neural-Trainer-Pro

# 2. Install Node.js dependencies
npm install

# 3. Setup Python virtual environment
python -m venv .venv

# Activate environment (choose your OS):
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# 4. Install PyTorch with CUDA support (for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5. Install model export dependencies
pip install onnx

# 6. Launch the application
npm run dev
```

> **Troubleshooting**: If PowerShell blocks script execution, use: `cmd /c "npm run dev"`

The application will open automatically at `http://localhost:5173` 🚀

## 🎮 Usage Workflow

1. **📤 Dataset Upload** - Drag and drop your image dataset (automatically detects structure)
2. **⚙️ Configure Training** - Set epochs, batch size, and learning rate
3. **📊 Monitor Progress** - Watch real-time metrics with interactive charts
4. **💾 Auto-Save Models** - Models saved to `server/trainedModel/` after each training
5. **📥 Export Models** - Download in PyTorch, ONNX, or TorchScript
6. **📜 Review History** - Analyze all past training sessions with pagination

### Model Storage & Cleanup
- All trained models are automatically saved to `server/trainedModel/`
- **Auto-Cleanup**: When you export a model, the system automatically downloads the converted file and then **deletes both the converted file and the original model** from the server to save space.

### Advanced Monitoring
Access detailed training insights through browser developer tools for:
- Real-time status updates and system information
- Per-epoch performance metrics and timing
- Dataset composition and validation results
- Training completion summaries and diagnostics

## 📁 Project Structure

```
Neural-Trainer-Pro/
├── src/                          # Frontend React application
│   ├── components/
│   │   ├── DatasetSelector.jsx  # Dataset upload interface
│   │   ├── TrainingConfig.jsx   # Hyperparameter controls
│   │   ├── TrainingProgress.jsx # Real-time monitoring
│   │   ├── TrainingHistory.jsx  # Session management
│   │   ├── ModelExport.jsx      # Export interface
│   │   └── Sidebar.jsx          # Navigation
│   ├── ml/
│   │   ├── model.js             # Neural network architecture
│   │   └── training.js          # Training orchestration
│   ├── utils/
│   │   ├── charts.js            # Chart configurations
│   │   └── storage.js           # Data persistence
│   └── App.jsx                  # Main application
├── server/                       # Backend Node.js services
│   ├── server.js                # Express API & WebSocket
│   ├── python/                  # Python ML scripts
│   │   ├── train.py             # PyTorch training pipeline
│   │   ├── export_to_onnx.py    # ONNX export
│   │   ├── export_to_torchscript.py # TorchScript export
│   │   ├── export_to_coreml.py  # CoreML export
│   │   └── export_to_tflite.py  # TFLite export
│   ├── uploads/                 # Dataset storage
│   └── trainedModel/            # Model repository
└── Configuration files (Vite, Tailwind, etc.)
```

## Supported Dataset Formats

The platform automatically detects and adapts to multiple dataset structures:

| Format | Structure | Use Case |
|--------|-----------|----------|
| **Class Folders** | Images organized by class directories | Custom datasets, rapid prototyping |
| **Train/Test Split** | Separate training and testing directories | Pre-split datasets, competitions |
| **CIFAR-10 Binary** | Binary batch files | Benchmarking, standardized testing |
| **Flat Structure** | All images in single directory | Single-class tasks, preprocessing |

## Configuration

### Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **Epochs** | 10 | 1-100 | Training iterations over complete dataset |
| **Batch Size** | 32 | 8-128 | Memory usage and training speed trade-off |
| **Learning Rate** | 0.001 | 0.0001-0.01 | Convergence speed and stability balance |

### Model Architecture

The platform uses an optimized convolutional neural network (SimpleCNN) with progressive feature extraction and regularization. Architecture can be customized by modifying `server/python/train.py`.

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

