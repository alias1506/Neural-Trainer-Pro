import sys
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path
import time
from datetime import datetime

class YOLODataset(Dataset):
    """Dataset loader for YOLO format (images + labels folders)"""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Load class names
        self._load_classes()
        # Load images from YOLO structure
        self._load_images()
    
    def _load_classes(self):
        # Try to find classes.txt
        classes_file = self.root / 'classes.txt'
        if not classes_file.exists():
            classes_file = self.root / 'obj.names'
        
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        
        if not self.class_names:
            self.class_names = ['object']  # Default single class
    
    def _load_images(self):
        # Look for images in train/valid folders
        image_dirs = []
        for split in ['train', 'valid', 'test']:
            split_dir = self.root / split / 'images'
            if split_dir.exists():
                image_dirs.append(split_dir)
        
        # If no split dirs, look for images directly
        if not image_dirs:
            img_dir = self.root / 'images'
            if img_dir.exists():
                image_dirs.append(img_dir)
            else:
                # Look for any images in root
                image_dirs.append(self.root)
        
        # Collect all images
        for img_dir in image_dirs:
            for img_path in img_dir.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    self.images.append(str(img_path))
                    
                    # Try to find corresponding label file
                    label_path = self._get_label_path(img_path)
                    if label_path and label_path.exists():
                        label = self._parse_yolo_label(label_path)
                        self.labels.append(label)
                    else:
                        # Default to class 0 if no label found
                        self.labels.append(0)
    
    def _get_label_path(self, img_path):
        # Convert images/xxx.jpg to labels/xxx.txt
        img_path = Path(img_path)
        label_dir = img_path.parent.parent / 'labels'
        label_path = label_dir / (img_path.stem + '.txt')
        return label_path
    
    def _parse_yolo_label(self, label_path):
        # Read YOLO label file (format: class_id x y w h)
        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    return int(parts[0])  # Return class ID
        except:
            pass
        return 0  # Default to class 0
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.images[idx]
        image = Image.open(img_path)
        # Handle palette images with transparency
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        image = image.convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CIFARBinaryDataset(Dataset):
    """Dataset loader for CIFAR-10/100 binary format (.bin files)"""
    def __init__(self, root, transform=None, is_cifar100=False):
        self.root = Path(root)
        self.transform = transform
        self.is_cifar100 = is_cifar100
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Load class names
        self._load_classes()
        # Load binary data
        self._load_binary_data()
    
    def _load_classes(self):
        # Try to find class names file
        for filename in ['batches.meta.txt', 'classes.txt', 'labels.txt']:
            classes_file = self.root / filename
            if classes_file.exists():
                with open(classes_file, 'r', encoding='utf-8', errors='ignore') as f:
                    self.class_names = [line.strip() for line in f if line.strip()]
                if self.class_names:
                    return
        
        # Default CIFAR-10 classes
        if not self.class_names:
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                               'dog', 'frog', 'horse', 'ship', 'truck']
    
    def _load_binary_data(self):
        import pickle
        import numpy as np
        
        # Look for binary batch files
        batch_files = []
        for pattern in ['data_batch_*', 'train_batch_*', 'batch_*']:
            batch_files.extend(sorted(self.root.glob(pattern)))
        
        if not batch_files:
            # Look for any .bin files or pickled files
            batch_files = list(self.root.glob('*batch*'))
        
        print(f"Found {len(batch_files)} batch files: {[f.name for f in batch_files]}", file=sys.stderr)
        
        for batch_file in batch_files:
            try:
                # First, try to load as pickle (Python CIFAR format)
                with open(batch_file, 'rb') as f:
                    batch_dict = pickle.load(f, encoding='bytes')
                    
                    # Handle both string and bytes keys
                    data_key = b'data' if b'data' in batch_dict else 'data'
                    labels_key = b'labels' if b'labels' in batch_dict else 'labels'
                    
                    if labels_key not in batch_dict:
                        # Try alternative key names
                        labels_key = b'fine_labels' if b'fine_labels' in batch_dict else 'fine_labels'
                    
                    batch_data = batch_dict[data_key]
                    batch_labels = batch_dict[labels_key]
                    
                    # CIFAR format: data is (N, 3072) for CIFAR-10/100
                    # Reshape to (N, 3, 32, 32) then transpose to (N, 32, 32, 3)
                    num_images = len(batch_labels)
                    batch_data = batch_data.reshape(num_images, 3, 32, 32).transpose(0, 2, 3, 1)
                    
                    self.images.extend(batch_data)
                    self.labels.extend(batch_labels)
                    
                    print(f"Loaded {num_images} images from {batch_file.name} (pickle format)", file=sys.stderr)
            except (pickle.UnpicklingError, KeyError) as e:
                # If pickle fails, try raw binary format (C/C++ CIFAR binary)
                try:
                    print(f"Trying raw binary format for {batch_file.name}...", file=sys.stderr)
                    with open(batch_file, 'rb') as f:
                        data = f.read()
                    
                    # CIFAR-10 binary format: each record is 1 label byte + 3072 image bytes
                    # CIFAR-100 binary format: 2 label bytes (coarse + fine) + 3072 image bytes
                    record_size = 3073 if not self.is_cifar100 else 3074
                    num_records = len(data) // record_size
                    
                    if num_records == 0:
                        print(f"File {batch_file.name} is too small for binary format", file=sys.stderr)
                        continue
                    
                    for i in range(num_records):
                        offset = i * record_size
                        record = data[offset:offset + record_size]
                        
                        if len(record) < record_size:
                            break
                        
                        # Extract label (first byte for CIFAR-10)
                        label = record[0]
                        
                        # Extract image data (3072 bytes = 32x32x3)
                        img_data = np.frombuffer(record[1:3073], dtype=np.uint8)
                        
                        # Reshape from (3072,) to (3, 32, 32) then transpose to (32, 32, 3)
                        img_data = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
                        
                        self.images.append(img_data)
                        self.labels.append(label)
                    
                    print(f"Loaded {num_records} images from {batch_file.name} (raw binary format)", file=sys.stderr)
                except Exception as e2:
                    print(f"Failed to load {batch_file.name} as raw binary: {e2}", file=sys.stderr)
                    continue
        
        # Convert to numpy arrays
        if self.images:
            import numpy as np
            self.images = np.array(self.images)
            self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np
        
        # Get image data (already in HWC format, values 0-255)
        img_data = self.images[idx]
        
        # Convert to PIL Image
        image = Image.fromarray(img_data.astype(np.uint8))
        label = int(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class FlatImageDataset(Dataset):
    """Dataset loader for flat structure with classes.txt"""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Load class names from classes.txt
        self._load_classes()
        # Load all images and infer labels from filenames or folder structure
        self._load_images()
    
    def _load_classes(self):
        # Try common class file names first
        common_names = ['classes.txt', 'obj.names', 'labels.txt', 'batches.meta.txt', 'class_names.txt']
        
        for filename in common_names:
            classes_file = self.root / filename
            if classes_file.exists():
                with open(classes_file, 'r', encoding='utf-8', errors='ignore') as f:
                    self.class_names = [line.strip() for line in f if line.strip()]
                if self.class_names:
                    return
        
        # If not found, search for ANY .txt file in root and check if it contains class names
        if not self.class_names:
            for txt_file in self.root.glob('*.txt'):
                try:
                    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        # Check if it looks like a class list (reasonable number of short lines)
                        if 2 <= len(lines) <= 1000 and all(len(line) < 100 for line in lines[:10]):
                            self.class_names = lines
                            print(f"Found classes in {txt_file.name}: {self.class_names[:5]}...", file=sys.stderr)
                            return
                except:
                    continue
        
        if not self.class_names:
            self.class_names = ['class_0']  # Default single class
    
    def _load_images(self):
        # Look in common locations
        search_paths = [
            self.root,
            self.root / 'images',
            self.root / 'train' / 'images',
            self.root / 'valid' / 'images',
            self.root / 'test' / 'images'
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for img_path in search_path.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    self.images.append(str(img_path))
                    
                    # Try to get label from corresponding .txt file (YOLO style)
                    label = self._get_label_from_file(img_path)
                    if label is None:
                        # Default to class 0
                        label = 0
                    
                    self.labels.append(label)
    
    def _get_label_from_file(self, img_path):
        # Check for label file in ../labels/ directory
        img_path = Path(img_path)
        
        # Try different label locations
        label_paths = [
            img_path.parent.parent / 'labels' / (img_path.stem + '.txt'),
            img_path.parent / (img_path.stem + '.txt'),
        ]
        
        for label_path in label_paths:
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            parts = line.split()
                            class_id = int(parts[0])
                            return class_id
                except:
                    pass
        
        return None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.images[idx]
        image = Image.open(img_path)
        # Handle palette images with transparency
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        image = image.convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Load images from folder structure
        self._load_images()
    
    def _load_images(self):
        # Check for train/test structure
        if (self.root / 'train').exists():
            data_dir = self.root / 'train'
        else:
            data_dir = self.root
        
        # Get class folders
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        self.class_names = sorted([d.name for d in class_dirs])
        
        for class_idx, class_dir in enumerate(sorted(class_dirs)):
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.images[idx]
        image = Image.open(img_path)
        # Handle palette images with transparency
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        image = image.convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

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

def send_progress(data):
    """Send progress update to Node.js"""
    print(f"PROGRESS:{json.dumps(data)}", flush=True)

def train_model(dataset_path, config):
    # Configuration
    epochs = config.get('epochs', 10)
    batch_size = config.get('batchSize', 32)
    learning_rate = config.get('learningRate', 0.001)
    dataset_type = config.get('datasetType', 'auto')  # Get dataset type from config
    
    # Data augmentation for training (prevent overfitting)
    train_transform = transforms.Compose([
        transforms.Resize((32, 32), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((32, 32), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset based on type
    train_dataset = None
    val_dataset = None
    num_classes = 0
    dataset_root = Path(dataset_path)
    
    # Load based on dataset type from config (optimized for speed)
    send_progress({'type': 'status', 'message': f'Initializing {dataset_type} dataset...'})
    
    # First, load dataset to get info and split indices
    if dataset_type == 'yolo':
        full_dataset = YOLODataset(dataset_path, transform=val_transform)
        num_classes = len(full_dataset.class_names)
        classes = full_dataset.class_names
    
    elif dataset_type in ['cifar-binary', 'cifar-10', 'cifar-100']:
        # Check if binary format
        if any(dataset_root.glob('*batch*')) or any(dataset_root.glob('*.bin')):
            is_cifar100 = 'cifar-100' in dataset_type.lower()
            full_dataset = CIFARBinaryDataset(dataset_path, transform=val_transform, is_cifar100=is_cifar100)
            num_classes = len(full_dataset.class_names)
            classes = full_dataset.class_names
        else:
            # Standard ImageFolder
            folder_path = dataset_root / 'train' if (dataset_root / 'train').exists() else dataset_root
            full_dataset = datasets.ImageFolder(folder_path, transform=val_transform)
            num_classes = len(full_dataset.classes)
            classes = full_dataset.classes
    
    elif dataset_type in ['train-test-split', 'class-folders']:
        # Standard ImageFolder structure
        folder_path = dataset_root / 'train' if (dataset_root / 'train').exists() else dataset_root
        full_dataset = datasets.ImageFolder(folder_path, transform=val_transform)
        num_classes = len(full_dataset.classes)
        classes = full_dataset.classes
    
    else:
        # Auto-detect (fallback only)
        if (dataset_root / 'train').exists():
            full_dataset = datasets.ImageFolder(dataset_root / 'train', transform=val_transform)
            num_classes = len(full_dataset.classes)
            classes = full_dataset.classes
        else:
            full_dataset = datasets.ImageFolder(dataset_root, transform=val_transform)
            num_classes = len(full_dataset.classes)
            classes = full_dataset.classes
    
    send_progress({
        'type': 'info',
        'classes': classes,
        'numClasses': num_classes,
        'numImages': len(full_dataset)
    })
    
    if len(full_dataset) == 0:
        send_progress({'type': 'error', 'message': 'No images found in dataset!'})
        sys.exit(1)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size]
    )
    
    # Create separate datasets with different transforms
    if dataset_type == 'yolo':
        train_dataset_full = YOLODataset(dataset_path, transform=train_transform)
        val_dataset_full = YOLODataset(dataset_path, transform=val_transform)
    elif dataset_type in ['cifar-binary', 'cifar-10', 'cifar-100']:
        if any(dataset_root.glob('*batch*')) or any(dataset_root.glob('*.bin')):
            is_cifar100 = 'cifar-100' in dataset_type.lower()
            train_dataset_full = CIFARBinaryDataset(dataset_path, transform=train_transform, is_cifar100=is_cifar100)
            val_dataset_full = CIFARBinaryDataset(dataset_path, transform=val_transform, is_cifar100=is_cifar100)
        else:
            folder_path = dataset_root / 'train' if (dataset_root / 'train').exists() else dataset_root
            train_dataset_full = datasets.ImageFolder(folder_path, transform=train_transform)
            val_dataset_full = datasets.ImageFolder(folder_path, transform=val_transform)
    else:
        folder_path = dataset_root / 'train' if (dataset_root / 'train').exists() else dataset_root
        if not folder_path.exists():
            folder_path = dataset_root
        train_dataset_full = datasets.ImageFolder(folder_path, transform=train_transform)
        val_dataset_full = datasets.ImageFolder(folder_path, transform=val_transform)
    
    # Apply the split indices
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices.indices)
    
    # Optimize DataLoader for GPU if available
    use_cuda = torch.cuda.is_available()
    dataloader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': use_cuda,  # Faster data transfer to GPU
        'num_workers': 0  # Set to 0 to avoid multiprocessing pickle errors on Windows during cancellation
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    
    send_progress({'type': 'status', 'message': f'Training on {train_size} images, validating on {val_size}'})
    
    # Model setup - Automatically detect and use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        device_info = f'GPU: {gpu_name} ({gpu_memory:.1f}GB)'
        send_progress({'type': 'device', 'device': device_info})
    else:
        device = torch.device('cpu')
        send_progress({'type': 'device', 'device': 'CPU (No GPU detected)'})
    
    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # Add weight decay (L2 regularization) to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Non-blocking transfer for GPU (faster data loading)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Send real-time updates every 10 batches or at the last batch
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                current_train_loss = running_loss / (batch_idx + 1)
                current_train_acc = correct / total
                elapsed = time.time() - start_time
                
                # Send batch progress (not final epoch update)
                send_progress({
                    'type': 'batch',
                    'epoch': epoch + 1,
                    'totalEpochs': epochs,
                    'batch': batch_idx + 1,
                    'totalBatches': num_batches,
                    'trainLoss': current_train_loss,
                    'trainAcc': current_train_acc,
                    'valLoss': 0,  # Will be updated after validation
                    'valAcc': 0,
                    'elapsed': elapsed
                })
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        elapsed = time.time() - start_time
        
        # Send progress
        send_progress({
            'type': 'epoch',
            'epoch': epoch + 1,
            'totalEpochs': epochs,
            'trainLoss': train_loss,
            'trainAcc': train_acc,
            'valLoss': val_loss,
            'valAcc': val_acc,
            'elapsed': elapsed
        })
    
    # Save model with datetime in filename in trainedModel folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'TrainedModel_{timestamp}.pth'
    # Ensure model is saved in trainedModel folder (outside uploads)
    models_dir = os.path.join(os.path.dirname(__file__), 'trainedModel')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_filename)
    
    # Save model with metadata including num_classes for conversion
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'classes': classes,
        'timestamp': timestamp
    }, model_path)
    
    # Clear GPU memory if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    send_progress({
        'type': 'complete',
        'modelPath': model_path,
        'finalTrainAcc': train_acc,
        'finalValAcc': val_acc,
        'numClasses': num_classes,
        'classes': classes
    })

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python train.py <dataset_path> <config_json>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    config = json.loads(sys.argv[2])
    
    try:
        train_model(dataset_path, config)
    except Exception as e:
        send_progress({'type': 'error', 'message': str(e)})
        sys.exit(1)
