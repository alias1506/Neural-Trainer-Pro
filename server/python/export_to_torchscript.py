import sys
import os
import torch
import torch.nn as nn
import json
import traceback


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


def load_pytorch_model(model_path, num_classes=10):
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
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
        return model
    except Exception as e:
        raise Exception(f"Failed to load PyTorch model: {str(e)}")


if __name__ == '__main__':
    try:
        if len(sys.argv) < 3:
            print(json.dumps({'success': False, 'error': 'Usage: python export_to_torchscript.py <model.pth> <output.pt> [num_classes]'}))
            sys.exit(1)
        
        model_path = sys.argv[1]
        output_path = sys.argv[2]
        num_classes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        
        model = load_pytorch_model(model_path, num_classes)
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
        
        print(json.dumps({'success': True, 'output_path': output_path}))
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}))
        sys.exit(1)