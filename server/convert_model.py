"""
Model Conversion Script
Converts PyTorch models to different formats (ONNX, TorchScript, etc.)
"""
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
import traceback

class SimpleCNN(nn.Module):
    """Simple CNN model for image classification - MUST MATCH train.py"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
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
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def convert_to_onnx(model_path, output_path, num_classes=10):
    """Convert PyTorch model to ONNX format"""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle both old format (state_dict) and new format (with metadata)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Use num_classes from checkpoint if available
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
    else:
        state_dict = checkpoint
    
    model = SimpleCNN(num_classes)
    model.load_state_dict(state_dict)
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
    """Convert PyTorch model to TorchScript format"""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle both old format (state_dict) and new format (with metadata)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
    else:
        state_dict = checkpoint
    
    model = SimpleCNN(num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save TorchScript model
    traced_model.save(output_path)
    return output_path

def convert_to_coreml(model_path, output_path, num_classes=10):
    """Convert PyTorch model to CoreML format (requires coremltools)"""
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("coremltools is required for CoreML export. Install with: pip install coremltools")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle both old format (state_dict) and new format (with metadata)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
    else:
        state_dict = checkpoint
    
    model = SimpleCNN(num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=(1, 3, 32, 32), name="input")],
        outputs=[ct.TensorType(name="output")]
    )
    
    # Save CoreML model
    mlmodel.save(output_path)
    return output_path

def convert_to_tflite(model_path, output_path, num_classes=10):
    """Convert PyTorch model to TensorFlow Lite (requires onnx-tf)"""
    import tempfile
    import os
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError:
        raise ImportError("onnx, onnx-tf, and tensorflow are required. Install with: pip install onnx onnx-tf tensorflow")
    
    # First convert to ONNX
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        onnx_path = tmp.name
    
    try:
        convert_to_onnx(model_path, onnx_path, num_classes)
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Export as TensorFlow saved model
        with tempfile.TemporaryDirectory() as tmp_dir:
            tf_rep.export_graph(tmp_dir)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
        
        return output_path
    finally:
        # Clean up temporary ONNX file
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)

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
