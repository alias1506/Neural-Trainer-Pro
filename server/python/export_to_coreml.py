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
            print(json.dumps({'success': False, 'error': 'Usage: python export_to_coreml.py <model.pth> <output.mlmodel> [num_classes]'}))
            sys.exit(1)
        
        model_path = sys.argv[1]
        output_path = sys.argv[2]
        num_classes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        
        try:
            import coremltools as ct
        except ImportError:
            print(json.dumps({'success': False, 'error': 'CoreML conversion requires: pip install coremltools'}))
            sys.exit(1)
        
        model = load_pytorch_model(model_path, num_classes)
        model.eval()
        
        # Create example input
        example_input = torch.randn(1, 3, 32, 32)
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input", shape=example_input.shape)],
            convert_to="mlprogram"  # Use ML Program format (newer)
        )
        
        # Save the model
        mlmodel.save(output_path)
        
        print(json.dumps({'success': True, 'output_path': output_path}))
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}))
        sys.exit(1)
