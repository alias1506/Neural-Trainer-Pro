"""
Standalone TFLite Converter
Converts PyTorch models to TensorFlow Lite format via ONNX
Usage: python export_to_tflite.py <model.pth> <output.tflite> [num_classes]
"""

import sys
import json
import os
from pathlib import Path


def convert_pth_to_tflite(pth_path, output_path, num_classes):
    """Convert PyTorch .pth model to TFLite format"""
    
    # Import check with detailed error reporting
    try:
        from google.protobuf import runtime_version
    except ImportError as e:
        return {
            "success": False,
            "error": f"Protobuf runtime_version missing: {e}",
            "hint": "Run: pip install --force-reinstall 'protobuf>=5.27.0,<6.0'"
        }
    
    try:
        import torch
        import torch.nn as nn
        import onnx
        import onnx2tf
        import tensorflow as tf
    except ImportError as e:
        return {
            "success": False,
            "error": f"Missing dependency: {e}",
            "hint": "Install with: pip install onnx2tf tensorflow"
        }
    
    try:
        # Validate input file
        if not os.path.exists(pth_path):
            return {"success": False, "error": f"Model file not found: {pth_path}"}
        
        # Load PyTorch model with SimpleCNN architecture
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=10, use_batchnorm=True, hidden_size=256):
                super(SimpleCNN, self).__init__()
                self.use_batchnorm = use_batchnorm
                
                if use_batchnorm:
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
                        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.3)
                    )
                    self.classifier = nn.Sequential(
                        nn.Linear(128*4*4, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(hidden_size, num_classes)
                    )
                else:
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
                    )
                    self.classifier = nn.Sequential(
                        nn.Identity(), nn.Linear(128*4*4, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(hidden_size, num_classes)
                    )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier[1:](x) if not self.use_batchnorm else self.classifier(x)
                return x
        
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
            if 'num_classes' in checkpoint:
                num_classes = checkpoint['num_classes']
        else:
            state_dict = checkpoint
        
        has_bn = any('running_mean' in k for k in state_dict.keys())
        hidden = state_dict['classifier.1.weight'].shape[0] if 'classifier.1.weight' in state_dict else 256
        
        model = SimpleCNN(num_classes, has_bn, hidden)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Create paths
        base_path = Path(output_path).with_suffix('')
        onnx_path = str(base_path) + '.onnx'
        saved_model_dir = str(base_path) + '_saved_model'
        
        # Export to ONNX
        dummy_input = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Convert ONNX to SavedModel
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
            output_signaturedefs=True,
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True
        )
        
        # Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Cleanup intermediate files
        try:
            Path(onnx_path).unlink(missing_ok=True)
            import shutil
            if os.path.exists(saved_model_dir):
                shutil.rmtree(saved_model_dir)
        except Exception:
            pass  # Ignore cleanup errors
        
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        return {
            "success": True,
            "output_path": output_path,
            "size_mb": round(model_size, 2)
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    if len(sys.argv) < 4:
        result = {
            "success": False,
            "error": "Usage: python export_to_tflite.py <model.pth> <output.tflite> <num_classes>"
        }
    else:
        pth_path = sys.argv[1]
        output_path = sys.argv[2]
        num_classes = int(sys.argv[3])
        
        result = convert_pth_to_tflite(pth_path, output_path, num_classes)
    
    print(json.dumps(result))
    sys.exit(0 if result.get("success") else 1)
