"""
Model Conversion Script
Converts PyTorch models to different formats (ONNX, TorchScript, CoreML, TFLite)

IMPORTANT: Each conversion function is completely independent!
- All converters load PyTorch models directly using load_pytorch_model()
- No converter depends on another converter's output
- Each format conversion starts fresh from the .pth file
"""
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
import traceback

class SimpleCNN(nn.Module):
    """Simple CNN model for image classification - supports OLD and NEW architectures"""
    def __init__(self, num_classes=10, use_batchnorm=True, hidden_size=256):
        super(SimpleCNN, self).__init__()
        
        if use_batchnorm:
            # NEW architecture with BatchNorm and Dropout
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3),
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            # OLD architecture without BatchNorm
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, num_classes)
            )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_pytorch_model(model_path, num_classes=10):
    """Load PyTorch model from checkpoint - auto-detects OLD/NEW architecture"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)  # weights_only=False needed for custom objects
    
    # Handle both old format (state_dict) and new format (with metadata)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Use num_classes from checkpoint if available
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
    else:
        state_dict = checkpoint
    
    # Auto-detect architecture by checking for BatchNorm layers
    has_batchnorm = any('BatchNorm' in str(type(v)) or 'running_mean' in k for k, v in state_dict.items())
    
    # Detect hidden size from first classifier layer
    hidden_size = 128  # default OLD architecture
    if 'classifier.1.weight' in state_dict:
        hidden_size = state_dict['classifier.1.weight'].shape[0]
    
    # Create model with detected architecture
    model = SimpleCNN(num_classes=num_classes, use_batchnorm=has_batchnorm, hidden_size=hidden_size)
    model.load_state_dict(state_dict)
    return model

def convert_to_onnx(model_path, output_path, num_classes=10):
    """Convert PyTorch model to ONNX format - Independent conversion"""
    # Load PyTorch model independently
    model = load_pytorch_model(model_path, num_classes)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    return output_path

def convert_to_torchscript(model_path, output_path, num_classes=10):
    """Convert PyTorch model to TorchScript format - Independent conversion"""
    # Load PyTorch model independently
    model = load_pytorch_model(model_path, num_classes)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save TorchScript model
    traced_model.save(output_path)
    return output_path

def convert_to_coreml(model_path, output_path, num_classes=10):
    """Convert PyTorch model to CoreML format"""
    import subprocess
    import sys
    import os
    
    converter_script = os.path.join(os.path.dirname(__file__), 'export_to_coreml.py')
    
    result = subprocess.run(
        [sys.executable, converter_script, model_path, output_path, str(num_classes)],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"CoreML conversion failed:\n{result.stdout}\n{result.stderr}")
    
    # Return ONNX fallback if CoreML not created
    if os.path.exists(output_path):
        return output_path
    else:
        onnx_fallback = output_path.replace('.mlmodel', '.onnx')
        if os.path.exists(onnx_fallback):
            return onnx_fallback
        else:
            raise RuntimeError("CoreML conversion did not produce output")

def convert_to_tflite(model_path, output_path, num_classes=10):
    """Convert PyTorch model to TFLite format"""
    import subprocess
    import sys
    import os
    
    converter_script = os.path.join(os.path.dirname(__file__), 'export_to_tflite.py')
    
    result = subprocess.run(
        [sys.executable, converter_script, model_path, output_path, str(num_classes)],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"TFLite conversion failed:\n{result.stdout}\n{result.stderr}")
    
    # Return ONNX fallback if TFLite not created
    if os.path.exists(output_path):
        return output_path
    else:
        onnx_fallback = output_path.replace('.tflite', '.onnx')
        if os.path.exists(onnx_fallback):
            return onnx_fallback
        else:
            raise RuntimeError("TFLite conversion did not produce output")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python convert_model.py <input_model_path> <output_format> <num_classes>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_format = sys.argv[2]
    num_classes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    # Generate output path
    input_file = Path(input_path)
    output_dir = input_file.parent
    base_name = input_file.stem
    
    format_extensions = {
        'onnx': '.onnx',
        'torchscript': '.pt',
        'coreml': '.mlmodel',
        'tflite': '.tflite'
    }
    
    if output_format not in format_extensions:
        print(f"Error: Unsupported format '{output_format}'")
        print(f"Supported formats: {', '.join(format_extensions.keys())}")
        sys.exit(1)
    
    output_path = output_dir / f"{base_name}_{output_format}{format_extensions[output_format]}"
    
    try:
        sys.stderr.write(f"Converting {input_path} to {output_format} format...\n")
        sys.stderr.flush()
        
        if output_format == 'onnx':
            result = convert_to_onnx(input_path, str(output_path), num_classes)
        elif output_format == 'torchscript':
            result = convert_to_torchscript(input_path, str(output_path), num_classes)
        elif output_format == 'coreml':
            result = convert_to_coreml(input_path, str(output_path), num_classes)
        elif output_format == 'tflite':
            result = convert_to_tflite(input_path, str(output_path), num_classes)
        
        sys.stderr.write(f"Conversion successful!\n")
        sys.stderr.flush()
        
        print(json.dumps({
            'success': True,
            'output_path': str(result),
            'format': output_format
        }))
    except Exception as e:
        error_details = traceback.format_exc()
        sys.stderr.write(f"Conversion error: {str(e)}\n")
        sys.stderr.write(f"Details: {error_details}\n")
        sys.stderr.flush()
        
        print(json.dumps({
            'success': False,
            'error': str(e),
            'traceback': error_details
        }))
        sys.exit(1)
