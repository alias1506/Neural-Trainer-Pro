# Setup Script for Neural Trainer Pro

Write-Host "=== Neural Trainer Pro Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Node.js
Write-Host "Checking Node.js..." -ForegroundColor Yellow
$nodeVersion = node --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Node.js not found. Please install Node.js 16+ from https://nodejs.org/" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Node.js $nodeVersion found" -ForegroundColor Green

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Python not found. Please install Python 3.8+ from https://python.org/" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python $pythonVersion found" -ForegroundColor Green

Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Cyan

# Install frontend dependencies
Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Frontend installation failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Frontend dependencies installed" -ForegroundColor Green

# Install server dependencies
Write-Host "Installing server dependencies..." -ForegroundColor Yellow
cd server
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Server installation failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Server dependencies installed" -ForegroundColor Green

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Python installation failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python dependencies installed" -ForegroundColor Green

cd ..

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "1. Terminal 1: cd server && npm start" -ForegroundColor White
Write-Host "2. Terminal 2: npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "Then open: http://localhost:5173" -ForegroundColor Yellow
