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

## Overview

Neural Trainer Pro is a production-ready machine learning training platform that seamlessly integrates PyTorch's computational power with a modern React interface. Deploy custom image classification models with enterprise-level visualization, automated workflows, and multi-format export capabilities.

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
| **Frontend** | React, Vite, Material-UI, Chart.js, Tailwind CSS |
| **Backend** | Node.js, Express, WebSocket |
| **ML Engine** | PyTorch, ONNX, CoreML, TensorFlow |
| **Infrastructure** | GPU/CUDA Support, Virtual Environments |

## Getting Started

### Prerequisites
- Node.js 16 or higher
- Python 3.11 or higher
- CUDA-capable GPU (optional, recommended for performance)

### Installation

```bash
# Clone repository
git clone https://github.com/alias1506/Neural-Trainer-Pro.git
cd Neural-Trainer-Pro

# Install dependencies
npm install

# Setup Python environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install ML dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnx coremltools tensorflow onnx-tf

# Launch application
npm run dev
```

> **Windows Users**: If script execution is restricted, use `cmd /c "npm run dev"`

The application will automatically launch in your default browser.

## Usage Workflow

1. **Dataset Upload** - Import your organized image dataset via drag-and-drop interface
2. **Configure Parameters** - Customize training hyperparameters to suit your needs
3. **Monitor Training** - Track real-time metrics with interactive visualizations
4. **Export Models** - Download trained models in your preferred deployment format
5. **Review History** - Analyze past training sessions and performance metrics

### Advanced Monitoring
Access detailed training insights through browser developer tools for:
- Real-time status updates and system information
- Per-epoch performance metrics and timing
- Dataset composition and validation results
- Training completion summaries and diagnostics

## Architecture

```
Neural-Trainer-Pro/
├── src/                 # Frontend application
│   ├── components/      # React UI components
│   ├── ml/             # Model definitions
│   └── utils/          # Utility functions
├── server/             # Backend services
│   ├── server.js       # API & WebSocket server
│   ├── train.py        # Training pipeline
│   └── convert_model.py # Format conversion
└── ...                 # Configuration files
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
- **Chart.js** - Data visualization
- **Material-UI** - Component library
- **React** - Frontend framework

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
