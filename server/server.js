const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const WebSocket = require('ws');

const app = express();
const PORT = 3001;

// WebSocket server for real-time updates
const wss = new WebSocket.Server({ port: 3002 });
let wsClient = null;
let currentTrainingProcess = null; // Track current training process

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  wsClient = ws;
});

// Middleware
app.use(cors());
app.use(express.json());

// Simple storage - just save to temp directory first
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const tempDir = path.join(__dirname, 'uploads', 'temp');
    await fs.mkdir(tempDir, { recursive: true });
    cb(null, tempDir);
  },
  filename: (req, file, cb) => {
    // Keep the original name (file_X_filename)
    cb(null, file.originalname);
  }
});

// Detect dataset structure
async function buildFolderTree(dirPath, maxDepth = 10, currentDepth = 0) {
  if (currentDepth >= maxDepth) {
    console.log(`  Max depth ${maxDepth} reached for ${dirPath}`);
    return [];
  }
  
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    const children = [];
    
    // Removed verbose logging
    
    for (const entry of entries) {
      try {
        const fullPath = path.join(dirPath, entry.name);
        
        if (entry.isDirectory()) {
          const subChildren = await buildFolderTree(fullPath, maxDepth, currentDepth + 1);
          children.push({
            name: entry.name,
            path: fullPath,
            isFolder: true,
            children: subChildren || []
          });
        } else if (entry.isFile()) {
          children.push({
            name: entry.name,
            path: fullPath,
            isFolder: false
          });
        }
      } catch (entryError) {
        // Ignore entry errors
      }
    }
    
    // Tree building complete
    
    return children;
  } catch (error) {
    // Ignore directory errors
    return [];
  }
}

async function getClassNames(datasetPath) {
  // Try to find and read class names from various files
  const classFiles = ['batches.meta.txt', 'classes.txt', 'labels.txt', 'obj.names', 'class_names.txt'];
  
  for (const filename of classFiles) {
    const filePath = path.join(datasetPath, filename);
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const classes = content.split('\n').map(line => line.trim()).filter(line => line.length > 0);
      if (classes.length > 0) {
        return classes;
      }
    } catch (e) {
      // File doesn't exist, continue
    }
  }
  
  // If no class file found, try to get folder names as classes
  try {
    const entries = await fs.readdir(datasetPath, { withFileTypes: true });
    const folders = entries.filter(e => e.isDirectory()).map(e => e.name);
    
    // Check if we have train/test structure
    if (folders.includes('train')) {
      const trainPath = path.join(datasetPath, 'train');
      const trainEntries = await fs.readdir(trainPath, { withFileTypes: true });
      const classFolders = trainEntries.filter(e => e.isDirectory()).map(e => e.name);
      if (classFolders.length > 0) {
        return classFolders;
      }
    }
    
    // Otherwise use root folders as classes (excluding common dataset folders)
    const excludeFolders = ['train', 'test', 'val', 'valid', 'images', 'labels'];
    const classFolders = folders.filter(f => !excludeFolders.includes(f.toLowerCase()));
    if (classFolders.length > 0) {
      return classFolders;
    }
  } catch (e) {
    // Ignore class folder errors
  }
  
  return [];
}

async function detectDatasetStructure(datasetPath) {
  const entries = await fs.readdir(datasetPath, { withFileTypes: true });
  const folders = entries.filter(e => e.isDirectory()).map(e => e.name);
  const files = entries.filter(e => e.isFile()).map(e => e.name);
  
  // Get class names
  const classes = await getClassNames(datasetPath);
  
  // Build folder tree
  const treeChildren = await buildFolderTree(datasetPath, 10, 0);
  
  const tree = {
    name: path.basename(datasetPath),
    path: datasetPath,
    isFolder: true,
    children: treeChildren
  };
  
  // Helper function to check if a folder contains class subfolders with images
  async function hasClassSubfolders(folderPath) {
    try {
      const subEntries = await fs.readdir(folderPath, { withFileTypes: true });
      const subFolders = subEntries.filter(e => e.isDirectory());
      
      if (subFolders.length === 0) return false;
      
      // Check if at least one subfolder contains images
      for (const subFolder of subFolders) {
        const subFolderPath = path.join(folderPath, subFolder.name);
        const subFiles = await fs.readdir(subFolderPath, { withFileTypes: true });
        const hasImages = subFiles.some(f => f.isFile() && /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(f.name));
        if (hasImages) return true;
      }
      return false;
    } catch (e) {
      return false;
    }
  }
  
  // Check for common structures
  const hasTrainTest = folders.includes('train') && folders.includes('test');
  const hasTrainFolder = folders.includes('train');
  const hasTestFolder = folders.includes('test');
  const hasValFolder = folders.includes('val') || folders.includes('valid');
  const hasClassFolders = folders.length > 0 && folders.every(f => !['train', 'test', 'val', 'valid'].includes(f.toLowerCase()));
  const hasBinFiles = files.some(f => f.endsWith('.bin'));
  const hasImageFiles = files.some(f => /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(f));
  const hasCsvFiles = files.some(f => f.endsWith('.csv'));
  

  
  // CIFAR-10/100 specific files
  const hasCifarBatch = files.some(f => /data_batch_\d+/.test(f) || f === 'test_batch' || f === 'batches.meta');
  const hasCifarMeta = files.some(f => f === 'batches.meta.txt' || f === 'batches.meta');
  const hasCifarHtml = files.some(f => f === 'batches.meta.html');
  
  // YOLO format detection (check for yaml config or labels structure)
  const hasYoloConfig = files.some(f => f === 'data.yaml' || f === 'dataset.yaml' || f.endsWith('.yaml'));
  const hasClassesTxt = files.some(f => f === 'classes.txt' || f === 'obj.names' || f === 'class.txt');
  const hasLabelsFolder = folders.includes('labels') || folders.includes('annotations');
  const hasImagesFolder = folders.includes('images');
  const lowerFolders = folders.map(f => f.toLowerCase());
  const hasTrainValidFolders = lowerFolders.includes('train') && 
                                (lowerFolders.includes('valid') || lowerFolders.includes('val'));
  
  // Check for YOLO structure: train/val folders with images and corresponding txt labels
  async function hasYoloStructure() {
    // Check if train/val/test folders contain images with corresponding .txt files
    const dataFolders = folders.filter(f => ['train', 'val', 'valid', 'test', 'images'].includes(f.toLowerCase()));
    
    if (dataFolders.length === 0) return false;
    
    for (const folder of dataFolders) {
      const folderPath = path.join(datasetPath, folder);
      try {
        const folderFiles = await fs.readdir(folderPath, { withFileTypes: true });
        const folderFileNames = folderFiles.filter(f => f.isFile()).map(f => f.name);
        
        // Check if there are corresponding txt labels for images
        const hasImageFile = folderFileNames.some(f => /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(f));
        const hasTxtFile = folderFileNames.some(f => f.endsWith('.txt') && f !== 'classes.txt');
        
        if (hasImageFile && hasTxtFile) {
          // Verify at least one image has a corresponding label
          for (const file of folderFileNames) {
            if (/\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(file)) {
              const baseName = file.replace(/\.(jpg|jpeg|png|gif|bmp|webp)$/i, '');
              if (folderFileNames.includes(baseName + '.txt')) {
                return true;
              }
            }
          }
        }
      } catch (e) {
        // Ignore errors
      }
    }
    return false;
  }
  
  const hasYoloLabels = await hasYoloStructure();
  

  
  // Dataset structure detected
  
  // CIFAR binary format (check FIRST before YOLO)
  if (hasBinFiles && (hasCifarBatch || hasCifarMeta || hasCifarHtml)) {
    return { 
      type: 'cifar-binary', 
      structure: 'CIFAR-10/100 binary format',
      supportedTasks: ['image-classification'],
      classes,
      numClasses: classes.length,
      tree
    };
  }
  // YOLO format detection (check BEFORE train-test-split since YOLO often has train folder)
  if (hasYoloConfig || hasYoloLabels || (hasLabelsFolder && hasImagesFolder) || 
      (hasTrainValidFolders && hasClassesTxt) || (hasTrainValidFolders && hasYoloLabels)) {

    return { 
      type: 'yolo', 
      structure: 'YOLO format (images + txt labels)',
      supportedTasks: ['object-detection', 'segmentation'],
      classes,
      numClasses: classes.length,
      tree
    };
  } 
  // CSV dataset detection
  else if (hasCsvFiles) {
    return { 
      type: 'csv', 
      structure: 'CSV dataset (tabular data)',
      supportedTasks: ['classification', 'regression'],
      classes,
      numClasses: classes.length,
      tree
    };
  } 
  // Train/Test split structure
  if (hasTrainTest) {
    const trainHasClasses = await hasClassSubfolders(path.join(datasetPath, 'train'));
    if (trainHasClasses) {
      return { 
        type: 'train-test-split', 
        structure: 'Organized train/test folders with class subfolders',
        supportedTasks: ['image-classification'],
        classes,
        numClasses: classes.length,
        tree
      };
    }
  } 
  // Only train folder (will use as class folders or split)
  else if (hasTrainFolder) {
    const trainHasClasses = await hasClassSubfolders(path.join(datasetPath, 'train'));
    if (trainHasClasses) {
      return { 
        type: 'train-test-split', 
        structure: 'Train folder with class subfolders (will auto-split validation)',
        supportedTasks: ['image-classification'],
        classes,
        numClasses: classes.length,
        tree
      };
    }
  }
  // Simple class folders
  if (hasClassFolders && folders.length > 0) {
    // Verify that folders actually contain images
    let hasValidClassFolders = false;
    for (const folder of folders) {
      const folderPath = path.join(datasetPath, folder);
      try {
        const folderEntries = await fs.readdir(folderPath, { withFileTypes: true });
        const folderHasImages = folderEntries.some(f => f.isFile() && /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(f.name));
        if (folderHasImages) {
          hasValidClassFolders = true;
          break;
        }
      } catch (e) {
        // Ignore read errors
      }
    }
    
    if (hasValidClassFolders) {
      return { 
        type: 'class-folders', 
        structure: 'Class folders (will auto-split 80/20 train/test)',
        supportedTasks: ['image-classification'],
        classes,
        numClasses: classes.length,
        tree
      };
    }
  } 
  // Flat image directory
  else if (hasImageFiles) {
    return { 
      type: 'flat-images', 
      structure: 'Flat image directory (single class or needs labeling)',
      supportedTasks: ['image-classification', 'clustering'],
      classes,
      numClasses: classes.length,
      tree
    };
  } 
  // Unknown/custom format
  else {
    return { 
      type: 'custom',
      classes,
      numClasses: classes.length, 
      structure: 'Custom format - will attempt auto-detection',
      supportedTasks: ['auto-detect'],
      tree
    };
  }
}

const upload = multer({ 
  storage,
  limits: {
    fileSize: 1024 * 1024 * 1024, // 1GB per file
  }
});

// Multer config to accept both files and pathMapping
const uploadMixed = multer({ storage, limits: { fileSize: 1024 * 1024 * 1024 } }).fields([
  { name: 'files' },
  { name: 'pathMapping', maxCount: 1 }
]);

// Upload dataset endpoint - unlimited files
app.post('/api/upload-dataset', async (req, res, next) => {
  
  // Use custom middleware to extract pathMapping BEFORE multer processes files
  uploadMixed(req, res, async (err) => {
    if (err) {
      // Multer error handled
      return res.status(500).json({ error: err.message });
    }
    
    // Extract pathMapping from uploaded file
    if (req.files && req.files.pathMapping && req.files.pathMapping[0]) {
      try {
        const pathMappingFile = req.files.pathMapping[0];
        const pathMappingContent = await fs.readFile(pathMappingFile.path, 'utf8');
        req.pathMapping = JSON.parse(pathMappingContent);
        
        // Delete the pathMapping file as we don't need it anymore
        await fs.unlink(pathMappingFile.path);
      } catch (e) {
        // Failed to parse pathMapping
      }
    }
    
    // Now handle the actual file uploads
    req.files = req.files.files || [];
    next();
  });
}, async (req, res) => {
  
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No files uploaded' });
    }
    
    if (!req.pathMapping) {
      return res.status(400).json({ error: 'Path mapping not found' });
    }
    

    
    // Clean up old datasets before uploading new one (keep only 1 dataset)
    const uploadsDir = path.join(__dirname, 'uploads');
    try {
      const entries = await fs.readdir(uploadsDir, { withFileTypes: true });
      for (const entry of entries) {
        // Skip temp folder only
        if (entry.name === 'temp') continue;
        
        const fullPath = path.join(uploadsDir, entry.name);
        try {
          if (entry.isDirectory()) {
            await fs.rm(fullPath, { recursive: true, force: true });
          } else if (entry.isFile() && entry.name.endsWith('.pth')) {
            await fs.unlink(fullPath);
          }
        } catch (delErr) {
          // Ignore errors
        }
      }
    } catch (cleanErr) {
      // Ignore errors
    }
    
    // Get the first file's full path to extract root folder name
    const firstFileKey = req.files[0].originalname;
    const firstFullPath = req.pathMapping[firstFileKey];
    const rootFolderName = firstFullPath.split(/[/\\]/)[0];
    const finalUploadDir = path.join(__dirname, 'uploads', rootFolderName);
    
    // Move and reorganize each file
    let movedCount = 0;
    for (const file of req.files) {
      let fileKey = file.originalname;
      let fullPath = req.pathMapping[fileKey];
      
      // If exact match not found, try to find by decoding or matching the actual filename part
      if (!fullPath) {
        // Try to find by matching the filename portion after the index
        const filenamePart = fileKey.replace(/^file_\d+_/, '');
        const matchingKey = Object.keys(req.pathMapping).find(key => {
          const keyFilename = key.replace(/^file_\d+_/, '');
          return keyFilename === filenamePart || decodeURIComponent(keyFilename) === filenamePart;
        });
        
        if (matchingKey) {
          fullPath = req.pathMapping[matchingKey];
        }
      }
      
      if (!fullPath) {
        // Silently skip files without path mapping (likely encoding issues with special characters)
        continue;
      }
      
      const pathParts = fullPath.split(/[/\\]/);
      const filename = pathParts[pathParts.length - 1];
      const subdirs = pathParts.slice(1, -1);
      
      const finalDir = subdirs.length > 0
        ? path.join(finalUploadDir, ...subdirs)
        : finalUploadDir;
      
      await fs.mkdir(finalDir, { recursive: true });
      
      const finalPath = path.join(finalDir, filename);
      await fs.rename(file.path, finalPath);
      
      movedCount++;
    }
    
    // Clean up temp directory
    const tempDir = path.join(__dirname, 'uploads', 'temp');
    try {
      await fs.rmdir(tempDir);
    } catch (e) {
      // Ignore if dir not empty or doesn't exist
    }
    
    const uploadPath = finalUploadDir;
    
    // Detect dataset structure
    const structure = await detectDatasetStructure(uploadPath);
    
    res.json({
      success: true,
      datasetPath: uploadPath,
      fileCount: req.files.length,
      structure
    });
  } catch (error) {
    // Upload error
    
    // Handle multer-specific errors
    if (error.code === 'LIMIT_FILE_COUNT') {
      return res.status(400).json({ 
        error: `Too many files. Maximum allowed is ${error.limit}. Please reduce the number of files.` 
      });
    }
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ 
        error: 'File too large. Please reduce file size.' 
      });
    }
    if (error.code === 'LIMIT_UNEXPECTED_FILE') {
      return res.status(400).json({ 
        error: 'Unexpected file field. Please ensure files are uploaded correctly.' 
      });
    }
    
    res.status(500).json({ error: error.message });
  }
});

// Error handling middleware for multer
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    // Multer error details omitted
    
    return res.status(400).json({ 
      error: `Upload error: ${error.message}`,
      code: error.code,
      filesProcessed: req.files ? req.files.length : 0
    });
  }
  next(error);
});

// Start training endpoint
app.post('/api/train', async (req, res) => {
  try {
    const { datasetPath, config } = req.body;
    
    // Kill any existing training process
    if (currentTrainingProcess) {
      // Killing existing training process
      currentTrainingProcess.kill('SIGTERM');
      currentTrainingProcess = null;
    }
    
    // Spawn Python process using virtual environment
    const pythonScript = path.join(__dirname, 'python', 'train.py');
    const pythonPath = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');
    const python = spawn(pythonPath, [
      pythonScript,
      datasetPath,
      JSON.stringify(config)
    ]);
    
    // Store the process reference
    currentTrainingProcess = python;
    
    let output = '';
    
    python.stdout.on('data', (data) => {
      const message = data.toString();
      output += message;
      // Python message received
      
      // Parse progress and send via WebSocket
      if (wsClient && wsClient.readyState === WebSocket.OPEN) {
        try {
          const lines = message.split('\n');
          lines.forEach(line => {
            if (line.startsWith('PROGRESS:')) {
              const progress = JSON.parse(line.replace('PROGRESS:', ''));
              wsClient.send(JSON.stringify(progress));
            }
          });
        } catch (e) {
          // Parse error
        }
      }
    });
    
    python.stderr.on('data', (data) => {
      // Python error
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        // Training completed
      } else if (code === null) {
        // Training cancelled
      } else {
        // Training failed
      }
      currentTrainingProcess = null;
    });
    
    res.json({ success: true, message: 'Training started' });
  } catch (error) {
    // Training error
    res.status(500).json({ error: error.message });
  }
});

// Cancel training endpoint
app.post('/api/cancel-training', async (req, res) => {
  try {
    if (currentTrainingProcess) {
      // Cancelling training
      
      // On Windows, use taskkill for forceful termination to avoid multiprocessing cleanup errors
      if (process.platform === 'win32') {
        const { exec } = require('child_process');
        exec(`taskkill /pid ${currentTrainingProcess.pid} /T /F`, (error) => {
          // Taskkill completed
        });
      } else {
        currentTrainingProcess.kill('SIGKILL');
      }
      
      currentTrainingProcess = null;
      
      // Send cancellation message via WebSocket
      if (wsClient && wsClient.readyState === WebSocket.OPEN) {
        wsClient.send(JSON.stringify({
          type: 'cancelled',
          message: 'Training cancelled by user'
        }));
      }
      
      res.json({ success: true, message: 'Training cancelled' });
    } else {
      res.json({ success: false, message: 'No training in progress' });
    }
  } catch (error) {
    // Cancel error
    res.status(500).json({ error: error.message });
  }
});

// Clean up uploads folder
app.post('/api/cleanup-uploads', async (req, res) => {
  try {
    const uploadsDir = path.join(__dirname, 'uploads');
    const entries = await fs.readdir(uploadsDir, { withFileTypes: true });
    
    let deletedCount = 0;
    for (const entry of entries) {
      // Skip temp folder
      if (entry.name === 'temp') continue;
      
      const fullPath = path.join(uploadsDir, entry.name);
      try {
        if (entry.isDirectory()) {
          await fs.rm(fullPath, { recursive: true, force: true });
          deletedCount++;
        } else if (entry.isFile() && entry.name.endsWith('.pth')) {
          await fs.unlink(fullPath);
          deletedCount++;
        }
      } catch (delErr) {
        // Failed to delete
      }
    }
    
    res.json({ success: true, message: `Cleaned up ${deletedCount} items` });
  } catch (error) {
    // Cleanup error
    res.status(500).json({ error: error.message });
  }
});

// Get dataset info
app.get('/api/dataset-info/:path', async (req, res) => {
  try {
    const datasetPath = decodeURIComponent(req.params.path);
    const structure = await detectDatasetStructure(datasetPath);
    res.json(structure);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Clean uploads folder endpoint
app.post('/api/clean-uploads', async (req, res) => {
  try {
    const uploadsDir = path.join(__dirname, 'uploads');
    const entries = await fs.readdir(uploadsDir, { withFileTypes: true });
    
    let deletedCount = 0;
    
    for (const entry of entries) {
      const fullPath = path.join(uploadsDir, entry.name);
      
      // Skip temp directory and current dataset
      if (entry.name === 'temp') continue;
      
      try {
        if (entry.isDirectory()) {
          // Delete directory recursively
          await fs.rm(fullPath, { recursive: true, force: true });
          deletedCount++;
        } else if (entry.isFile() && entry.name.endsWith('.pth')) {
          // Delete pretrained model files
          await fs.unlink(fullPath);
          deletedCount++;
        }
      } catch (error) {
        console.error(`Error deleting ${fullPath}:`, error);
      }
    }
    
    res.json({ success: true, deletedCount });
  } catch (error) {
    // Clean error
    res.status(500).json({ error: error.message });
  }
});

// Clean dataset only (keep models) - called after training completes
app.post('/api/clean-dataset', async (req, res) => {
  try {
    const uploadsDir = path.join(__dirname, 'uploads');
    const entries = await fs.readdir(uploadsDir, { withFileTypes: true });
    
    let deletedCount = 0;
    
    for (const entry of entries) {
      const fullPath = path.join(uploadsDir, entry.name);
      
      // Skip temp directory
      if (entry.name === 'temp') continue;
      
      try {
        // Delete all dataset directories and loose .pth files (not in Model folder)
        if (entry.isDirectory()) {
          await fs.rm(fullPath, { recursive: true, force: true });
          deletedCount++;
        } else if (entry.isFile() && entry.name.endsWith('.pth')) {
          // Delete loose .pth files in root uploads (not in Model folder)
          await fs.unlink(fullPath);
          deletedCount++;
        }
      } catch (error) {
        console.error(`Error deleting ${fullPath}:`, error);
      }
    }
    
    res.json({ success: true, deletedCount });
  } catch (error) {
    // Clean dataset error
    res.status(500).json({ error: error.message });
  }
});

// Get list of all trained models
app.get('/api/list-models', async (req, res) => {
  try {
    const modelsDir = path.join(__dirname, 'trainedModel');
    
    // Check if trainedModel directory exists
    try {
      await fs.access(modelsDir);
    } catch (error) {
      return res.json({ models: [] });
    }

    // Read all files in trainedModel directory
    const files = await fs.readdir(modelsDir);
    
    // Filter for .pth files and get their stats
    const models = [];
    for (const file of files) {
      if (file.endsWith('.pth')) {
        const filePath = path.join(modelsDir, file);
        const stats = await fs.stat(filePath);
        models.push({
          name: file,
          path: `trainedModel/${file}`,
          size: stats.size,
          createdAt: stats.birthtime,
          modifiedAt: stats.mtime
        });
      }
    }

    // Sort by creation date (newest first)
    models.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    // Auto-cleanup: If no models exist, remove all dataset folders
    if (models.length === 0) {
      try {
        const uploadsDir = path.join(__dirname, 'uploads');
        const entries = await fs.readdir(uploadsDir, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.isDirectory() && entry.name !== 'temp') {
            const fullPath = path.join(uploadsDir, entry.name);
            await fs.rm(fullPath, { recursive: true, force: true });
            console.log(`Auto-cleaned orphaned dataset: ${entry.name}`);
          }
        }
      } catch (cleanError) {
        console.warn('Auto-cleanup warning:', cleanError);
      }
    }

    res.json({ models });
  } catch (error) {
    // List models error
    res.status(500).json({ error: error.message });
  }
});

// Download model endpoint with format conversion
app.get('/api/download-model', async (req, res) => {
  try {
    const { modelPath, format = 'pytorch', numClasses = 10 } = req.query;
    
    if (!modelPath) {
      return res.status(400).json({ error: 'Model path is required' });
    }

    // Convert relative path to absolute path
    const absolutePath = path.isAbsolute(modelPath) 
      ? modelPath 
      : path.join(__dirname, modelPath);

    // Check if file exists
    try {
      await fs.access(absolutePath);
    } catch (error) {
      return res.status(404).json({ error: 'Model file not found', path: absolutePath });
    }

    let fileToDownload = absolutePath;
    let filename = path.basename(modelPath);

    // Convert model if format is not pytorch
    if (format !== 'pytorch' && format !== 'pth') {
      try {
        // Run conversion script using unified converter
        const pythonExe = path.resolve(__dirname, '..', '.venv', 'Scripts', 'python.exe');
        const convertScript = path.resolve(__dirname, 'python', 'convert_model.py');
        
        console.log('Python executable:', pythonExe);
        console.log('Convert script:', convertScript);
        console.log('Model path:', absolutePath);
        console.log('Format:', format);
        console.log('Num classes:', numClasses);
        
        const conversionProcess = spawn(pythonExe, [
          convertScript,
          absolutePath,
          format,
          numClasses.toString()
        ]);

        let outputData = '';
        let errorData = '';

        conversionProcess.stdout.on('data', (data) => {
          outputData += data.toString();
        });

        conversionProcess.stderr.on('data', (data) => {
          errorData += data.toString();
          console.error('Python stderr:', data.toString());
        });

        await new Promise((resolve, reject) => {
          conversionProcess.on('close', (code) => {
            if (code === 0) {
              try {
                const lines = outputData.trim().split('\n');
                const lastLine = lines[lines.length - 1];
                const result = JSON.parse(lastLine);
                if (result.success) {
                  fileToDownload = result.output_path;
                  filename = path.basename(fileToDownload);
                  resolve();
                } else {
                  const errorMsg = result.error || 'Conversion failed';
                  const details = result.traceback || errorData;
                  reject(new Error(`${errorMsg}\n\nDetails:\n${details}`));
                }
              } catch (e) {
                reject(new Error(`Failed to parse conversion result.\nOutput: ${outputData}\nError: ${errorData}`));
              }
            } else {
              reject(new Error(`Conversion process failed with code ${code}.\nOutput: ${outputData}\nError: ${errorData}`));
            }
          });
        });
      } catch (conversionError) {
        // Conversion error
        return res.status(500).json({ 
          error: 'Model conversion failed', 
          details: conversionError.message,
          hint: format === 'onnx' ? 'Install with: pip install onnx' :
                format === 'coreml' ? 'Install with: pip install coremltools' :
                format === 'tflite' ? 'Install with: pip install onnx onnx-tf tensorflow' :
                'Required dependencies may not be installed'
        });
      }
    }

    // Get file stats
    const stats = await fs.stat(fileToDownload);
    
    // Set appropriate headers for file download
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.setHeader('Content-Length', stats.size);

    // Stream the file
    const fileStream = require('fs').createReadStream(fileToDownload);
    fileStream.pipe(res);
    
    fileStream.on('error', (error) => {
      // File stream error
      if (!res.headersSent) {
        res.status(500).json({ error: 'Error streaming file' });
      }
    });

    // Clean up files after download
    fileStream.on('end', async () => {
      try {
        // Delete converted file if it exists
        if (fileToDownload !== absolutePath) {
          await fs.unlink(fileToDownload);
        }
        
        // Delete the original .pth model file after export
        await fs.unlink(absolutePath);
      } catch (e) {
        // Cleanup error
      }
    });

  } catch (error) {
    // Download error
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`WebSocket server running on ws://localhost:3002`);
});
