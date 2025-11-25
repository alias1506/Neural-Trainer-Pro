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

<div align="center">

**[Neural Trainer Pro](https://github.com/alias1506/Neural-Trainer-Pro)**

*Professional Machine Learning Training Platform*

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/alias1506/Neural-Trainer-Pro)

</div>

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
- **Universal Export** - Deploy to any platform with 5+ export formats
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
Export trained models to multiple industry-standard formats:
- **PyTorch** - Native deployment
- **ONNX** - Cross-platform inference
- **TorchScript** - Production optimization
- **CoreML** - iOS/macOS integration
- **TensorFlow Lite** - Mobile & edge devices

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
| **ML Engine** | PyTorch, ONNX, CoreML, TensorFlow Lite |
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

# 5. Install model export dependencies (optional)
pip install onnx coremltools tensorflow onnx-tf

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
5. **📥 Export Models** - Download in PyTorch, ONNX, TorchScript, CoreML, or TFLite
6. **📜 Review History** - Analyze all past training sessions with pagination

### Model Storage
- All trained models are automatically saved to `server/trainedModel/`
- Each model has a unique timestamp (e.g., `TrainedModel_20251120_113045.pth`)
- Models persist across sessions until explicitly exported and deleted
- Multiple models can be trained and stored simultaneously

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
│   │   ├── cifar.js             # CIFAR-10 handling
│   │   ├── onnx-export.js       # Model conversion
│   │   └── storage.js           # Data persistence
│   └── App.jsx                  # Main application
├── server/                       # Backend Node.js services
│   ├── server.js                # Express API & WebSocket
│   ├── train.py                 # PyTorch training pipeline
│   ├── convert_model.py         # Multi-format export
│   ├── uploads/                 # Dataset storage
│   └── trainedModel/            # Model repository
├── coverage/                     # Test coverage reports
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

## Core Capabilities

### Training Intelligence
- **Real-time Visualization**: Interactive performance charts with dual-axis metrics
- **Hardware Optimization**: Automatic GPU detection and utilization
- **Progress Tracking**: Live monitoring of training phases and timing
- **Historical Analysis**: Complete session history with comparative metrics

### Model Deployment
Convert trained models to industry-standard formats for seamless integration:
- **PyTorch** - Native framework deployment
- **ONNX** - Platform-agnostic inference
- **TorchScript** - Production-optimized execution
- **CoreML** - Apple ecosystem integration
- **TensorFlow Lite** - Mobile and embedded systems

### Data Management
- Intelligent dataset validation and preprocessing
- Automated class distribution analysis
- Post-training cleanup and optimization
- Visual dataset exploration tools

## Configuration

### Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **Epochs** | 10 | 1-100 | Training iterations over complete dataset |
| **Batch Size** | 32 | 8-128 | Memory usage and training speed trade-off |
| **Learning Rate** | 0.001 | 0.0001-0.01 | Convergence speed and stability balance |

### Model Architecture

The platform uses an optimized convolutional neural network with progressive feature extraction and regularization. Architecture can be customized by modifying the training pipeline configuration.

## Troubleshooting

### Common Issues

**Script Execution Restricted (Windows)**
```bash
# Solution: Use Command Prompt
cmd /c "npm run dev"

# Alternative: Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**GPU Not Detected**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python server/check_gpu.py
```

**Model Conversion Failures**
```bash
# Install conversion dependencies
pip install onnx coremltools tensorflow onnx-tf
```

**Python Version Compatibility**
```bash
# Use Python 3.11 for optimal compatibility
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnx coremltools tensorflow onnx-tf
```

**Connection Issues**
- Verify all services are running
- Check browser console for detailed diagnostics
- Ensure firewall allows necessary connections
- Restart application if issues persist

## Training Pipeline

The platform executes a sophisticated training workflow:

1. **Data Ingestion** - Automated dataset upload, validation, and structure analysis
2. **Initialization** - Model architecture setup with hardware optimization
3. **Training Loop** - Iterative learning with real-time performance monitoring
4. **Completion** - Model serialization and performance summary generation

## Best Practices

### Dataset Preparation
- Maintain balanced class distributions
- Ensure minimum 100 images per class for robust learning
- Use high-quality, well-lit images with consistent characteristics
- Remove corrupted or ambiguous samples

### Training Optimization
- Begin with default hyperparameters
- Monitor validation metrics to prevent overfitting
- Adjust learning rate based on convergence behavior
- Leverage GPU acceleration for optimal performance

### Performance Considerations
- GPU training provides 10-100x speedup over CPU
- Batch size impacts memory usage and training speed
- Regular validation prevents overfitting
- Historical comparison aids hyperparameter tuning

## Contributing

We welcome contributions from the community. Please follow these guidelines:

1. Fork the repository and create a feature branch
2. Implement changes with appropriate testing
3. Submit a pull request with detailed description
4. Ensure code adheres to project standards

## Acknowledgments

Built with industry-leading technologies:
- **PyTorch** - Deep learning framework
- **Chart.js** - Real-time data visualization
- **React** - Frontend framework
- **Tailwind CSS** - Utility-first CSS framework
- **SweetAlert2** - Beautiful alert dialogs

Special thanks to the open-source community for continuous innovation.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues, questions, or feature requests, please open a GitHub issue.

---

<div align="center">

**[Neural Trainer Pro](https://github.com/alias1506/Neural-Trainer-Pro)**

*Professional Machine Learning Training Platform*

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/alias1506/Neural-Trainer-Pro)

</div>
