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
import pandas as pd
import numpy as np

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
            self.class_names = ['class_0']  # Default single class
    
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
        
        # FAST: Just collect paths, don't load images or parse labels yet
        for img_dir in image_dirs:
            for img_path in img_dir.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    self.images.append(str(img_path))
                    self.labels.append(0)  # Default label, will parse on-demand if needed
    
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
        
        # Parse label on-demand (lazy loading)
        if self.labels[idx] == 0:  # Only if not already loaded
            label_path = self._get_label_path(Path(img_path))
            if label_path and label_path.exists():
                self.labels[idx] = self._parse_yolo_label(label_path)
        
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
            except (pickle.UnpicklingError, KeyError) as e:
                # If pickle fails, try raw binary format (C/C++ CIFAR binary)
                try:
                    with open(batch_file, 'rb') as f:
                        data = f.read()
                    
                    # CIFAR-10 binary format: each record is 1 label byte + 3072 image bytes
                    # CIFAR-100 binary format: 2 label bytes (coarse + fine) + 3072 image bytes
                    record_size = 3073 if not self.is_cifar100 else 3074
                    num_records = len(data) // record_size
                    
                    if num_records == 0:
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
                except Exception as e2:
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
                            return
                except:
                    continue
        
        if not self.class_names:
            self.class_names = ['class_0']  # Default single class
    
    def _load_images(self):
        # FAST: Just collect image paths, don't parse labels yet
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
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    self.images.append(str(img_path))
                    self.labels.append(0)  # Will parse on-demand during training
    
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
        
        # Parse label on-demand (lazy loading)
        if self.labels[idx] == 0:
            label = self._get_label_from_file(Path(img_path))
            if label is not None:
                self.labels[idx] = label
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CSVDataset(Dataset):
    """Dataset loader for CSV format (tabular data)"""
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        
        # Separate features and target
        # Use .copy() to avoid SettingWithCopyWarning
        X_df = self.df.iloc[:, :-1].copy()
        y_series = self.df.iloc[:, -1].copy()
        
        # Preprocess features: Handle categorical columns
        for col in X_df.columns:
            # Check if column is object (string) or category
            if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                # Convert to categorical codes
                X_df[col] = X_df[col].astype('category').cat.codes
        
        # Handle missing values (fill with 0)
        X_df = X_df.fillna(0)
        
        self.X = X_df.values.astype('float32')
        self.y = y_series.values
        
        # Encode labels if they are strings
        if self.y.dtype == 'object' or self.y.dtype.type is np.str_ or self.y.dtype.type is np.object_:
            self.classes = sorted(list(set(self.y)))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            self.y = np.array([self.class_to_idx[lbl] for lbl in self.y], dtype=np.int64)
        else:
            # Assume they are already integers 0..N-1 or similar
            self.classes = [str(i) for i in sorted(list(set(self.y)))]
            self.y = self.y.astype(np.int64)
            
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

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
    
    # Auto-configure validation split and early stopping if set to 0 (prevent overfitting)
    validation_split = config.get('validationSplit', 0)
    if validation_split == 0:
        validation_split = 0.2  # Automatic 20% validation split
        send_progress({'type': 'init', 'message': 'Using automatic 20% validation split to prevent overfitting'})
    
    early_stop_patience = config.get('patience', 0)
    if early_stop_patience == 0:
        early_stop_patience = 20  # Automatic patience of 20 epochs
        send_progress({'type': 'init', 'message': 'Using automatic early stopping (patience: 20 epochs) to prevent overfitting'})
    
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
    input_dim = 0
    dataset_root = Path(dataset_path)
    
    # Load based on dataset type from config (optimized for speed)
    send_progress({'type': 'init', 'message': f'Loading {dataset_type} dataset from {dataset_path}...'})
    
    # Load dataset ONCE and reuse it (FAST initialization with lazy loading)
    if dataset_type == 'yolo':
        # Load with minimal transform first (just to get metadata)
        train_dataset_full = YOLODataset(dataset_path, transform=train_transform)
        num_classes = len(train_dataset_full.class_names)
        classes = train_dataset_full.class_names
        total_samples = len(train_dataset_full)
        send_progress({'type': 'init', 'message': f'Found {total_samples} images in YOLO dataset'})
    
    elif dataset_type in ['cifar-binary', 'cifar-10', 'cifar-100']:
        # Check if binary format
        if any(dataset_root.glob('*batch*')) or any(dataset_root.glob('*.bin')):
            is_cifar100 = 'cifar-100' in dataset_type.lower()
            train_dataset_full = CIFARBinaryDataset(dataset_path, transform=train_transform, is_cifar100=is_cifar100)
            num_classes = len(train_dataset_full.class_names)
            classes = train_dataset_full.class_names
        else:
            # Standard ImageFolder
            folder_path = dataset_root / 'train' if (dataset_root / 'train').exists() else dataset_root
            train_dataset_full = datasets.ImageFolder(folder_path, transform=train_transform)
            num_classes = len(train_dataset_full.classes)
            classes = train_dataset_full.classes
        total_samples = len(train_dataset_full)
        send_progress({'type': 'init', 'message': f'Found {total_samples} images in CIFAR dataset'})
    
    elif dataset_type in ['train-test-split', 'class-folders']:
        # Standard ImageFolder structure
        folder_path = dataset_root / 'train' if (dataset_root / 'train').exists() else dataset_root
        train_dataset_full = datasets.ImageFolder(folder_path, transform=train_transform)
        num_classes = len(train_dataset_full.classes)
        classes = train_dataset_full.classes
        total_samples = len(train_dataset_full)
        send_progress({'type': 'init', 'message': f'Found {total_samples} images across {num_classes} classes'})
    
    elif dataset_type == 'csv':
        # Find CSV file
        csv_files = list(dataset_root.glob('*.csv'))
        if not csv_files:
             send_progress({'type': 'error', 'message': 'No CSV file found!'})
             sys.exit(1)
        
        csv_path = csv_files[0]
        train_dataset_full = CSVDataset(csv_path)
        num_classes = len(train_dataset_full.classes)
        classes = train_dataset_full.classes
        input_dim = train_dataset_full.X.shape[1]
        total_samples = len(train_dataset_full)
        
        # Disable image transforms for CSV data
        train_transform = None
        val_transform = None
        
        send_progress({'type': 'init', 'message': f'Loaded CSV dataset with {total_samples} samples, {input_dim} features, {num_classes} classes'})
    
    else:
        # Auto-detect (fallback only)
        if (dataset_root / 'train').exists():
            train_dataset_full = datasets.ImageFolder(dataset_root / 'train', transform=train_transform)
            num_classes = len(train_dataset_full.classes)
            classes = train_dataset_full.classes
        else:
            train_dataset_full = datasets.ImageFolder(dataset_root, transform=train_transform)
            num_classes = len(train_dataset_full.classes)
            classes = train_dataset_full.classes
        total_samples = len(train_dataset_full)
        send_progress({'type': 'init', 'message': f'Found {total_samples} images (auto-detected)'})
    
    send_progress({
        'type': 'info',
        'classes': classes,
        'numClasses': num_classes,
        'numImages': total_samples
    })
    
    if total_samples == 0:
        send_progress({'type': 'error', 'message': 'No images found in dataset!'})
        sys.exit(1)
    
    # Split dataset into train and validation using configured split ratio
    send_progress({'type': 'init', 'message': f'Splitting dataset ({int((1-validation_split)*100)}% train / {int(validation_split*100)}% validation)...'})
    train_size = int((1 - validation_split) * total_samples)
    val_size = total_samples - train_size
    
    # Create random split indices
    generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_samples), [train_size, val_size], generator=generator
    )
    
    # OPTIMIZATION: Reuse same dataset with transform wrapper instead of loading twice
    # This avoids scanning filesystem twice which is the slow part
    class TransformWrapper(Dataset):
        """Wrapper to apply different transforms to same dataset"""
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            # Get raw data without transform
            if hasattr(self.dataset, 'transform'):
                old_transform = self.dataset.transform
                self.dataset.transform = None
                img, label = self.dataset[idx]
                self.dataset.transform = old_transform
            else:
                img, label = self.dataset[idx]
            
            if self.transform:
                img = self.transform(img)
            return img, label
    
    val_dataset_full = TransformWrapper(train_dataset_full, val_transform)
    
    # Apply the split indices
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices.indices)
    
    # Optimize DataLoader for GPU if available
    send_progress({'type': 'init', 'message': 'Creating data loaders...'})
    use_cuda = torch.cuda.is_available()
    dataloader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': use_cuda,  # Faster data transfer to GPU
        'num_workers': 2 if use_cuda else 0,  # Use 2 workers for GPU, 0 for CPU
        'persistent_workers': True if use_cuda else False,  # Keep workers alive between epochs
        'prefetch_factor': 2 if use_cuda else None  # Prefetch batches for faster loading
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    
    send_progress({'type': 'init', 'message': f'Ready: {train_size} training images, {val_size} validation images'})
    
    # Model setup - Automatically detect and use GPU if available
    send_progress({'type': 'init', 'message': 'Initializing model and optimizer...'})
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        device_info = f'GPU: {gpu_name} ({gpu_memory:.1f}GB)'
        send_progress({'type': 'device', 'device': device_info})
    else:
        device = torch.device('cpu')
        send_progress({'type': 'device', 'device': 'CPU (No GPU detected)'})
    
    if dataset_type == 'csv':
        model = SimpleMLP(input_dim, num_classes).to(device)
    else:
        model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # Add weight decay (L2 regularization) to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Learning rate scheduler to reduce LR when validation loss plateaus (prevents overfitting)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Early stopping to prevent overfitting (patience configured above)
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    send_progress({'type': 'init', 'message': f'Starting training for {epochs} epochs...'})
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
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                elapsed = time.time() - start_time
                send_progress({
                    'type': 'epoch',
                    'epoch': epoch + 1,
                    'totalEpochs': epochs,
                    'trainLoss': train_loss,
                    'trainAcc': train_acc,
                    'valLoss': val_loss,
                    'valAcc': val_acc,
                    'elapsed': elapsed,
                    'earlyStop': True
                })
                break  # Stop training
        
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
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_filename = f'TrainedModel-{timestamp}.pth'
    # Ensure model is saved in server/trainedModel folder (not in python subfolder)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trainedModel')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_filename)
    
    # Save model with metadata including num_classes for conversion
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'classes': classes,
        'timestamp': timestamp,
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'train_loss': float(train_loss),
        'val_loss': float(val_loss)
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
