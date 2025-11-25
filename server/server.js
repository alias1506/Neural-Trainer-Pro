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
  wsClient = ws;
});

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

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

    return children;
  } catch (error) {
    // Ignore directory errors
    return [];
  }
}

async function getClassNames(datasetPath) {
  // Try to parse YAML file first (for YOLO datasets)
  const yamlFiles = ['data.yaml', 'dataset.yaml'];
  for (const yamlFile of yamlFiles) {
    const yamlPath = path.join(datasetPath, yamlFile);
    try {
      const content = await fs.readFile(yamlPath, 'utf-8');
      // Simple YAML parsing for 'names' field
      const namesMatch = content.match(/names:\s*\[([^\]]+)\]/);
      if (namesMatch) {
        const classes = namesMatch[1].split(',').map(c => c.trim().replace(/['"`]/g, ''));
        if (classes.length > 0) return classes;
      }
      // Also try list format
      const listMatch = content.match(/names:\s*\n((?:\s+-\s*.+\n?)+)/);
      if (listMatch) {
        const classes = listMatch[1].split('\n').map(line => line.trim().replace(/^-\s*/, '').replace(/['"`]/g, '')).filter(c => c.length > 0);
        if (classes.length > 0) return classes;
      }
    } catch (e) {
      // YAML file doesn't exist or parse error
    }
  }

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
        const folderEntries = await fs.readdir(folderPath, { withFileTypes: true });

        // Check if this folder has images/labels subdirectories (common YOLO structure)
        const subFolders = folderEntries.filter(e => e.isDirectory()).map(e => e.name.toLowerCase());
        if (subFolders.includes('images') && subFolders.includes('labels')) {
          // Check images folder for image files
          const imagesPath = path.join(folderPath, 'images');
          const imagesEntries = await fs.readdir(imagesPath, { withFileTypes: true });
          const hasImages = imagesEntries.some(f => f.isFile() && /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(f.name));

          // Check labels folder for txt files
          const labelsPath = path.join(folderPath, 'labels');
          const labelsEntries = await fs.readdir(labelsPath, { withFileTypes: true });
          const hasLabels = labelsEntries.some(f => f.isFile() && f.name.endsWith('.txt'));

          if (hasImages && hasLabels) {
            return true;
          }
        }

        // Also check if images and labels are directly in the folder (flat structure)
        const folderFileNames = folderEntries.filter(f => f.isFile()).map(f => f.name);

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

    // Clean up old datasets before uploading new one
    const uploadsDir = path.join(__dirname, 'uploads');
    await fs.mkdir(uploadsDir, { recursive: true }); // Ensure uploads dir exists

    try {
      const entries = await fs.readdir(uploadsDir, { withFileTypes: true });
      for (const entry of entries) {
        // Skip temp folder
        if (entry.name === 'temp') continue;

        const fullPath = path.join(uploadsDir, entry.name);
        try {
          if (entry.isDirectory()) {
            // On Windows, deleting and immediately recreating a folder can cause ENOENT
            // So if it's the 'dataset' folder, we'll just empty it
            if (entry.name === 'dataset') {
              const datasetEntries = await fs.readdir(fullPath);
              for (const subEntry of datasetEntries) {
                await fs.rm(path.join(fullPath, subEntry), { recursive: true, force: true });
              }
            } else {
              // Remove other directories (old named datasets)
              await fs.rm(fullPath, { recursive: true, force: true });
            }
          } else if (entry.isFile() && entry.name.endsWith('.pth')) {
            await fs.unlink(fullPath);
          }
        } catch (delErr) {
          console.error('Cleanup error for', fullPath, delErr);
        }
      }
    } catch (cleanErr) {
      console.error('Cleanup scan error:', cleanErr);
    }

    // Force a consistent dataset folder name to avoid ENOENT errors
    const rootFolderName = 'dataset';
    const finalUploadDir = path.join(__dirname, 'uploads', rootFolderName);

    // Ensure the final directory exists
    await fs.mkdir(finalUploadDir, { recursive: true });

    // Determine common root prefix from pathMapping
    let commonRoot = '';
    if (req.pathMapping) {
      const paths = Object.values(req.pathMapping);
      if (paths.length > 0) {
        const firstPath = paths[0];
        const parts = firstPath.split(/[/\\]/);
        if (parts.length > 1) {
          const potentialRoot = parts[0];
          const allShareRoot = paths.every(p => p.startsWith(potentialRoot + '/') || p.startsWith(potentialRoot + '\\'));
          if (allShareRoot) {
            commonRoot = potentialRoot;
          }
        }
      }
    }

    // Move files from temp to final destination
    for (const file of req.files) {
      const fileKey = file.originalname;
      const tempPath = path.join(__dirname, 'uploads', 'temp', fileKey);

      let destPath;

      // Check if it's a CSV file - put it directly in the root of dataset folder
      if (fileKey.toLowerCase().endsWith('.csv')) {
        destPath = path.join(finalUploadDir, fileKey);
      }
      // Handle folder uploads with path mapping
      else if (req.pathMapping && req.pathMapping[fileKey]) {
        let mappedPath = req.pathMapping[fileKey];

        // Strip common root if it exists
        if (commonRoot && (mappedPath.startsWith(commonRoot + '/') || mappedPath.startsWith(commonRoot + '\\'))) {
          mappedPath = mappedPath.substring(commonRoot.length + 1);
        }

        // Also strip 'dataset/' if it's still there (fallback)
        if (mappedPath.startsWith('dataset/') || mappedPath.startsWith('dataset\\')) {
          mappedPath = mappedPath.substring(8);
        }

        destPath = path.join(finalUploadDir, mappedPath);
      }
      // Fallback for other files
      else {
        destPath = path.join(finalUploadDir, fileKey);
      }

      // Ensure destination directory exists
      await fs.mkdir(path.dirname(destPath), { recursive: true });

      try {
        await fs.rename(tempPath, destPath);
      } catch (e) {
        // Fallback for cross-device moves
        await fs.copyFile(tempPath, destPath);
        await fs.unlink(tempPath);
      }
    }

    // Clean up temp directory
    try {
      await fs.rmdir(path.join(__dirname, 'uploads', 'temp'));
    } catch (e) { }

    const uploadPath = finalUploadDir;

    // Check if directory exists before detecting structure
    try {
      await fs.access(uploadPath);
    } catch (e) {
      throw new Error(`Upload failed: Destination directory ${uploadPath} does not exist.`);
    }

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

    console.log('\n=== Training Request Received ===');
    console.log(`Dataset: ${datasetPath}`);
    console.log(`Epochs: ${config.epochs}, Batch Size: ${config.batchSize}, Learning Rate: ${config.learningRate}`);

    // Kill any existing training process
    if (currentTrainingProcess) {
      console.log('Terminating existing training process...');
      currentTrainingProcess.kill('SIGTERM');
      currentTrainingProcess = null;
    }

    // Spawn Python process using virtual environment
    const pythonScript = path.join(__dirname, 'python', 'train.py');
    const pythonPath = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');
    console.log('Starting Python training process...');
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

      // Parse progress and send via WebSocket
      if (wsClient && wsClient.readyState === WebSocket.OPEN) {
        try {
          const lines = message.split('\n');
          lines.forEach(line => {
            if (line.startsWith('PROGRESS:')) {
              const progress = JSON.parse(line.replace('PROGRESS:', ''));

              // Log progress to server console
              if (progress.type === 'init') {
                console.log(`[INIT] ${progress.message}`);
              } else if (progress.type === 'epoch') {
                const elapsed = Math.floor(progress.elapsed);
                console.log(`[Epoch ${progress.epoch}/${progress.totalEpochs}] Loss: ${progress.trainLoss.toFixed(4)}, Acc: ${(progress.trainAcc * 100).toFixed(2)}%, Val Loss: ${progress.valLoss.toFixed(4)}, Val Acc: ${(progress.valAcc * 100).toFixed(2)}% (${elapsed}s)`);
                if (progress.earlyStop) {
                  console.log('Early stopping triggered!');
                }
              } else if (progress.type === 'complete') {
                console.log('\n=== Training Complete ===');
                console.log(`Final Train Acc: ${(progress.finalTrainAcc * 100).toFixed(2)}%, Val Acc: ${(progress.finalValAcc * 100).toFixed(2)}%`);
                console.log(`Model saved: ${progress.modelPath}`);
                console.log(`Classes (${progress.numClasses}): ${progress.classes.join(', ')}\n`);
              } else if (progress.type === 'error') {
                console.error(`[ERROR] ${progress.message}`);
              }

              wsClient.send(JSON.stringify(progress));
            }
          });
        } catch (e) {
          // Parse error
        }
      }
    });

    python.stderr.on('data', (data) => {
      const errorMsg = data.toString();
      console.error('[Python Error]', errorMsg);
    });

    python.on('close', (code) => {
      if (code === 0) {
        console.log('Training process completed successfully.\n');
      } else if (code === null) {
        console.log('Training process was cancelled.\n');
      } else {
        console.error(`Training process failed with exit code ${code}\n`);
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

    const pythonPath = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');

    // Read all files in trainedModel directory
    const files = await fs.readdir(modelsDir);

    const models = [];
    for (const file of files) {
      if (file.endsWith('.pth')) {
        const filePath = path.join(modelsDir, file);
        let stats;
        try {
          stats = await fs.stat(filePath);
        } catch (e) {
          continue; // Skip if can't read stats
        }

        // Parse date from filename if possible
        let modelDate = stats.mtime;
        const dateMatch = file.match(/TrainedModel-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})/);
        if (dateMatch) {
          const [y, m, d, h, min, s] = dateMatch[1].split('-');
          modelDate = new Date(y, m - 1, d, h, min, s);
        }

        models.push({
          id: file.replace('.pth', ''),
          name: file,
          path: `trainedModel/${file}`,
          sizeBytes: stats.size,
          date: modelDate.toISOString(),
          accuracy: null, // Will be populated by frontend from history
          metrics: null,  // Will be populated by frontend from history
          numClasses: null,
          classes: []
        });
      }
    }

    // Sort by creation date (newest first)
    models.sort((a, b) => new Date(b.date) - new Date(a.date));

    res.json({ models });
  } catch (error) {
    // List models error
    res.status(500).json({ error: error.message });
  }
});

// Delete individual model endpoint
// Delete individual model endpoint
app.delete('/api/delete-model/:modelId', async (req, res) => {
  try {
    const { modelId } = req.params;
    let modelFilename = modelId;

    // Ensure filename ends with .pth
    if (!modelFilename.endsWith('.pth')) {
      modelFilename += '.pth';
    }

    const modelPath = path.join(__dirname, 'trainedModel', modelFilename);

    // Check if model exists
    try {
      await fs.access(modelPath);
    } catch (error) {
      // Try without .pth just in case the ID was the full filename
      if (modelId.endsWith('.pth')) {
        const altPath = path.join(__dirname, 'trainedModel', modelId);
        try {
          await fs.access(altPath);
          // If found, proceed with this path
          await fs.unlink(altPath);
          return res.json({ success: true, message: `Model ${modelId} deleted successfully` });
        } catch (e) {
          // Both failed
          return res.status(404).json({ error: 'Model not found' });
        }
      }
      return res.status(404).json({ error: 'Model not found' });
    }

    // Delete the model file
    await fs.unlink(modelPath);

    res.json({ success: true, message: `Model ${modelFilename} deleted successfully` });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Clear all models endpoint
app.post('/api/clear-all-models', async (req, res) => {
  try {
    const modelsDir = path.join(__dirname, 'trainedModel');

    // Check if directory exists
    try {
      await fs.access(modelsDir);
    } catch (error) {
      return res.json({ success: true, message: 'No models to clear' });
    }

    // Read all files in trainedModel directory
    const files = await fs.readdir(modelsDir);
    let deletedCount = 0;

    // Delete all .pth files
    for (const file of files) {
      if (file.endsWith('.pth')) {
        const filePath = path.join(modelsDir, file);
        try {
          await fs.unlink(filePath);
          deletedCount++;
        } catch (e) {
          console.error(`Failed to delete ${file}:`, e);
        }
      }
    }

    // Also clear the uploads directory to fully reset
    const uploadsDir = path.join(__dirname, 'uploads');
    try {
      await fs.access(uploadsDir);
      const uploadFiles = await fs.readdir(uploadsDir);
      for (const file of uploadFiles) {
        await fs.unlink(path.join(uploadsDir, file));
      }
    } catch (e) {
      // Ignore if uploads dir doesn't exist or empty
    }
    res.json({ success: true, message: `Deleted ${deletedCount} model(s)`, deletedCount });
  } catch (error) {
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
        // Run conversion script
        const pythonExe = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');

        // Select the appropriate export script based on format
        let exportScript;
        let outputExt;
        switch (format) {
          case 'onnx':
            exportScript = path.join(__dirname, 'python', 'export_to_onnx.py');
            outputExt = '.onnx';
            break;
          case 'torchscript':
            exportScript = path.join(__dirname, 'python', 'export_to_torchscript.py');
            outputExt = '.pt';
            break;
          case 'coreml':
            exportScript = path.join(__dirname, 'python', 'export_to_coreml.py');
            outputExt = '.mlmodel';
            break;
          default:
            return res.status(400).json({ error: `Unsupported format: ${format}. Supported formats: pytorch, onnx, torchscript, coreml` });
        }

        // Create output path
        const outputPath = absolutePath.replace('.pth', outputExt);

        const conversionProcess = spawn(pythonExe, [
          exportScript,
          absolutePath,
          outputPath,
          numClasses.toString()
        ]);

        let outputData = '';
        let errorData = '';

        conversionProcess.stdout.on('data', (data) => {
          outputData += data.toString();
          console.log(`[Export ${format}] ${data.toString().trim()}`);
        });

        conversionProcess.stderr.on('data', (data) => {
          errorData += data.toString();
          console.error(`[Export ${format} Error] ${data.toString().trim()}`);
        });

        await new Promise((resolve, reject) => {
          conversionProcess.on('close', (code) => {
            console.log(`[Export ${format}] Process exited with code ${code}`);
            if (code === 0) {
              try {
                const lines = outputData.trim().split('\n');
                const lastLine = lines[lines.length - 1];
                console.log(`[Export ${format}] Parsing result: ${lastLine}`);
                const result = JSON.parse(lastLine);
                if (result.success) {
                  fileToDownload = result.output_path;
                  filename = path.basename(fileToDownload);
                  console.log(`[Export ${format}] Success! File: ${filename}`);
                  resolve();
                } else {
                  const errorMsg = result.error || 'Conversion failed';
                  const details = result.traceback || errorData;
                  console.error(`[Export ${format}] Failed: ${errorMsg}`);
                  reject(new Error(`${errorMsg}\n\nDetails:\n${details}`));
                }
              } catch (e) {
                console.error(`[Export ${format}] Parse error:`, e);
                reject(new Error(`Failed to parse conversion result.\nOutput: ${outputData}\nError: ${errorData}`));
              }
            } else {
              console.error(`[Export ${format}] Process failed with code ${code}`);
              reject(new Error(`Conversion process failed with code ${code}.\nOutput: ${outputData}\nError: ${errorData}`));
            }
          });
        });
      } catch (conversionError) {
        // Conversion error
        console.error(`[Export ${format}] Conversion error:`, conversionError);
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

    console.log(`[Download] Preparing to send file: ${filename} (${(await fs.stat(fileToDownload)).size} bytes)`);

    // Get file stats
    const stats = await fs.stat(fileToDownload);

    // Set CORS headers to allow download
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Expose-Headers', 'Content-Disposition');

    // Set appropriate headers for file download
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.setHeader('Content-Length', stats.size);
    res.setHeader('X-Content-Type-Options', 'nosniff');

    console.log(`[Download] Streaming file to client...`);

    // Stream the file
    const fileStream = require('fs').createReadStream(fileToDownload);
    fileStream.pipe(res);

    fileStream.on('end', () => {
      console.log(`[Download] File sent successfully: ${filename}`);

      // Clean up converted file after download
      if (fileToDownload !== absolutePath) {
        fs.unlink(fileToDownload)
          .then(() => console.log(`[Download] Cleaned up converted file: ${filename}`))
          .catch(err => console.error(`[Download] Cleanup error (converted):`, err));
      }

      // Also delete the original .pth file after export
      fs.unlink(absolutePath)
        .then(() => console.log(`[Download] Cleaned up original file: ${path.basename(absolutePath)}`))
        .catch(err => console.error(`[Download] Cleanup error (original):`, err));
    });

    fileStream.on('error', (error) => {
      // File stream error
      console.error(`[Download] Stream error:`, error);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Error streaming file' });
      }

      // Clean up on error
      if (fileToDownload !== absolutePath) {
        fs.unlink(fileToDownload).catch(() => { });
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
