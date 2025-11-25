import sys
import json
import torch
import torch.nn as nn
import traceback


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
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
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_pytorch_model(model_path, num_classes=10):
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            model.eval()
            return model
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
            if 'num_classes' in checkpoint:
                num_classes = checkpoint['num_classes']
        else:
            state_dict = checkpoint
        
        model = SimpleCNN(num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        return model
        
    except Exception as e:
        raise Exception(f"Failed to load PyTorch model: {str(e)}")


if __name__ == '__main__':
    try:
        # Set UTF-8 encoding for stdout/stderr to handle Unicode characters
        import sys
        if sys.platform == 'win32':
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        
        if len(sys.argv) < 3:
            print(json.dumps({'success': False, 'error': 'Usage: python export_to_tflite.py <model.pth> <output.tflite> [num_classes]'}))
            sys.exit(1)
        
        model_path = sys.argv[1]
        output_path = sys.argv[2]
        num_classes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        
        # First convert to ONNX, then to TFLite
        import onnx
        import tempfile
        import os
        
        model = load_pytorch_model(model_path, num_classes)
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        
        # Create temporary ONNX file
        temp_onnx = tempfile.NamedTemporaryFile(delete=False, suffix='.onnx')
        temp_onnx.close()
        
        try:
            # Export to ONNX first
            torch.onnx.export(
                model,
                dummy_input,
                temp_onnx.name,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                verbose=False  # Disable verbose output to avoid Unicode issues
            )
            
            # Convert ONNX to TFLite using onnx-tf and tensorflow
            try:
                import onnx_tf
                import tensorflow as tf
                
                onnx_model = onnx.load(temp_onnx.name)
                tf_rep = onnx_tf.backend.prepare(onnx_model)
                
                # Save as TensorFlow SavedModel
                temp_tf_dir = tempfile.mkdtemp()
                tf_rep.export_graph(temp_tf_dir)
                
                # Convert to TFLite
                converter = tf.lite.TFLiteConverter.from_saved_model(temp_tf_dir)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                
                # Save TFLite model
                with open(output_path, 'wb') as f:
                    f.write(tflite_model)
                
                print(json.dumps({'success': True, 'output_path': output_path}))
            except ImportError:
                print(json.dumps({'success': False, 'error': 'TFLite conversion requires: pip install onnx-tf tensorflow'}))
                sys.exit(1)
        finally:
            # Clean up temp file
            if os.path.exists(temp_onnx.name):
                os.unlink(temp_onnx.name)
        
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}))
        sys.exit(1)
