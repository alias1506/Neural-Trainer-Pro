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
        console.error(`Error processing entry ${entry.name}:`, entryError.message);
      }
    }
    
    // Tree building complete
    
    return children;
  } catch (error) {
    console.error(`Error reading directory ${dirPath}:`, error.message);
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
    console.error('Error reading class folders:', e);
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
  
  // Check for common structures
  const hasTrainTest = folders.includes('train') && folders.includes('test');
  const hasClassFolders = folders.length > 0 && folders.every(f => !['train', 'test', 'val'].includes(f));
  const hasBinFiles = files.some(f => f.endsWith('.bin'));
  const hasImageFiles = files.some(f => /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(f));
  const hasCsvFiles = files.some(f => f.endsWith('.csv'));
  
  // CIFAR-10/100 specific files
  const hasCifarBatch = files.some(f => /data_batch_\d+/.test(f) || f === 'test_batch' || f === 'batches.meta');
  const hasCifarMeta = files.some(f => f === 'batches.meta.txt' || f === 'batches.meta');
  const hasCifarHtml = files.some(f => f === 'batches.meta.html');
  
  // YOLO format detection (more specific - needs both images and corresponding txt labels)
  const hasYoloConfig = files.some(f => f === 'data.yaml' || f === 'dataset.yaml');
  const hasYoloLabels = files.some(f => {
    // YOLO labels are .txt files that correspond to image files
    if (!f.endsWith('.txt')) return false;
    const baseName = f.replace('.txt', '');
    return files.some(imgFile => {
      const imgBase = imgFile.replace(/\.(jpg|jpeg|png|gif|bmp|webp)$/i, '');
      return imgBase === baseName;
    });
  });
  
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
  // YOLO format detection
  else if ((hasYoloLabels && hasImageFiles) || hasYoloConfig) {
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
  else if (hasTrainTest) {
    return { 
      type: 'train-test-split', 
      structure: 'Organized train/test folders with class subfolders',
      supportedTasks: ['image-classification'],
      classes,
      numClasses: classes.length,
      tree
    };
  } 
  // Simple class folders
  else if (hasClassFolders) {
    return { 
      type: 'class-folders', 
      structure: 'Class folders (will auto-split 80/20 train/test)',
      supportedTasks: ['image-classification'],
      classes,
      numClasses: classes.length,
      tree
    };
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
      console.error('Multer error:', err);
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
        console.error('Failed to parse pathMapping file:', e.message);
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
        if (entry.name === 'temp') continue; // Skip temp folder
        
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
      const fileKey = file.originalname;
      const fullPath = req.pathMapping[fileKey];
      
      if (!fullPath) {
        console.warn(`Warning: No path mapping for ${fileKey}`);
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
    console.error('Upload error:', error);
    
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
    console.error('\n===== MULTER ERROR =====');
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
    console.error('Error field:', error.field);
    console.error('Files processed before error:', req.files ? req.files.length : 0);
    if (req.files && req.files.length > 0) {
      console.error('Last successful file:', req.files[req.files.length - 1].originalname);
    }
    console.error('========================\n');
    
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
      console.log('Killing existing training process...');
      currentTrainingProcess.kill('SIGTERM');
      currentTrainingProcess = null;
    }
    
    // Spawn Python process using virtual environment
    const pythonScript = path.join(__dirname, 'train.py');
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
      console.log('Python:', message);
      
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
          console.error('Parse error:', e);
        }
      }
    });
    
    python.stderr.on('data', (data) => {
      console.error('Python error:', data.toString());
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        console.log('Training completed successfully');
      } else if (code === null) {
        console.log('Training was cancelled');
      } else {
        console.error('Training failed with code:', code);
      }
      currentTrainingProcess = null;
    });
    
    res.json({ success: true, message: 'Training started' });
  } catch (error) {
    console.error('Training error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Cancel training endpoint
app.post('/api/cancel-training', async (req, res) => {
  try {
    if (currentTrainingProcess) {
      console.log('Cancelling training process...');
      currentTrainingProcess.kill('SIGTERM');
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
    console.error('Cancel error:', error);
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
    console.error('Clean error:', error);
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
      
      // Skip temp directory and model files
      if (entry.name === 'temp') continue;
      if (entry.isFile()) continue; // Keep all files (models)
      
      try {
        if (entry.isDirectory()) {
          // Delete dataset directories only
          await fs.rm(fullPath, { recursive: true, force: true });
          deletedCount++;
        }
      } catch (error) {
        console.error(`Error deleting ${fullPath}:`, error);
      }
    }
    
    res.json({ success: true, deletedCount });
  } catch (error) {
    console.error('Clean dataset error:', error);
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

    // Check if file exists
    try {
      await fs.access(modelPath);
    } catch (error) {
      return res.status(404).json({ error: 'Model file not found' });
    }

    let fileToDownload = modelPath;
    let filename = path.basename(modelPath);

    // Convert model if format is not pytorch
    if (format !== 'pytorch' && format !== 'pth') {
      try {
        // Run conversion script
        const pythonExe = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');
        const convertScript = path.join(__dirname, 'convert_model.py');
        
        const conversionProcess = spawn(pythonExe, [
          convertScript,
          modelPath,
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
        console.error('Conversion error details:', conversionError);
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
      console.error('File stream error:', error);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Error streaming file' });
      }
    });

    // Clean up files after download
    fileStream.on('end', async () => {
      try {
        // Delete converted file if it's not the original
        if (fileToDownload !== modelPath) {
          await fs.unlink(fileToDownload);
          console.log(`Deleted converted file: ${fileToDownload}`);
        }
        
        // Delete original trained model file after conversion
        // (except for pytorch format which IS the original)
        if (format !== 'pytorch' && format !== 'pth') {
          await fs.unlink(modelPath);
          console.log(`Deleted original model file after conversion: ${modelPath}`);
        }
      } catch (e) {
        console.error('Cleanup error:', e);
      }
    });

  } catch (error) {
    console.error('Download error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`WebSocket server running on ws://localhost:3002`);
});
