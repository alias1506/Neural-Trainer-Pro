"""
Standalone CoreML Converter
Converts PyTorch models to CoreML format via ONNX
Usage: python export_to_coreml.py <model.pth> <output.mlmodel> [num_classes]

Note: On Windows, creates ONNX file + instructions for macOS conversion
"""

import torch
import torch.nn as nn
import sys
import os


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


def load_model(path, num_classes=10):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
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
    return model, num_classes


if __name__ == '__main__':
    import json
    import traceback
    
    try:
        if len(sys.argv) < 3:
            print(json.dumps({'success': False, 'error': 'Usage: python export_to_coreml.py <model.pth> <output.mlmodel> [num_classes]'}))
            sys.exit(1)
        
        model, nc = load_model(sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
        output = sys.argv[2]
        onnx_path = output.replace('.mlmodel', '.onnx')
        
        # Step 1: Create ONNX
        torch.onnx.export(
            model, torch.randn(1, 3, 32, 32), onnx_path,
            export_params=True, opset_version=13,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        
        # Step 2: Try CoreML conversion
        try:
            import coremltools as ct
            import onnx
            
            onnx_model = onnx.load(onnx_path)
            mlmodel = ct.convert(onnx_model, source='onnx', minimum_deployment_target=ct.target.iOS15)
            mlmodel.save(output)
            print(json.dumps({'success': True, 'output_path': output}))
            
        except ImportError:
            readme = output.replace('.mlmodel', '_Instructions.txt')
            with open(readme, 'w') as f:
                f.write("CoreML Conversion Instructions\n")
                f.write("="*50 + "\n\n")
                f.write(f"ONNX file created: {os.path.basename(onnx_path)}\n\n")
                f.write("To convert on macOS:\n")
                f.write("1. pip install coremltools\n")
                f.write("2. Run:\n")
                f.write("   import coremltools as ct\n")
                f.write(f"   ct.convert('{os.path.basename(onnx_path)}', source='onnx').save('{os.path.basename(output)}')\n\n")
                f.write("Or use ONNX Runtime Mobile (iOS/Android support)\n")
            print(json.dumps({'success': False, 'error': 'coremltools not available on Windows', 'output_path': onnx_path, 'instructions': readme}))
        
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}))
        sys.exit(1)
