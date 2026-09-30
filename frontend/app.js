/**
 * MatteForge - AI Background Remover Frontend
 * Handles upload, API communication, preview, and download
 */

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const resultsArea = document.getElementById('resultsArea');
const originalCanvas = document.getElementById('originalCanvas');
const resultCanvas = document.getElementById('resultCanvas');
const outputCanvas = document.getElementById('outputCanvas');
const comparisonSlider = document.getElementById('comparisonSlider');
const downloadBtn = document.getElementById('downloadBtn');
const outputSize = document.getElementById('outputSize');

// State
let currentFile = null;
let currentResultBlob = null;
let currentResultUrl = null;

// Initialize
document.addEventListener('DOMContentLoaded', init);

function init() {
  setupDragAndDrop();
  setupFileInput();
  setupProcessButton();
  setupComparisonSlider();
  setupDownloadButton();
}

/**
 * Set up drag and drop functionality
 */
function setupDragAndDrop() {
  // Prevent default drag behaviors
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  // Highlight on drag over
  ['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, highlight, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, unhighlight, false);
  });

  // Handle dropped files
  uploadArea.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

function highlight() {
  uploadArea.classList.add('dragover');
}

function unhighlight() {
  uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length > 0) {
    handleFiles(files[0]);
  }
}

/**
 * Set up file input change handler
 */
function setupFileInput() {
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFiles(e.target.files[0]);
    }
  });

  // Click on upload area to trigger file input
  uploadArea.addEventListener('click', () => {
    fileInput.click();
  });
}

/**
 * Handle the selected/dropped file
 */
function handleFiles(file) {
  // Validate file type
  if (!file.type.startsWith('image/')) {
    showError('Please select a valid image file (JPEG, PNG, WEBP, etc.)');
    return;
  }

  // Validate file size (max 15MB)
  const maxSize = 15 * 1024 * 1024;
  if (file.size > maxSize) {
    showError('File too large. Maximum size is 15MB.');
    return;
  }

  currentFile = file;
  processBtn.disabled = false;

  // Display original image preview
  displayOriginalImage(file);
}

/**
 * Display the original image on the canvas
 */
function displayOriginalImage(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      // Set canvas dimensions
      originalCanvas.width = img.width;
      originalCanvas.height = img.height;
      resultCanvas.width = img.width;
      resultCanvas.height = img.height;
      outputCanvas.width = img.width;
      outputCanvas.height = img.height;

      // Draw original image
      const ctx = originalCanvas.getContext('2d');
      ctx.drawImage(img, 0, 0);

      // Clear result canvas
      const resultCtx = resultCanvas.getContext('2d');
      resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

      // Draw checkerboard pattern for transparency
      drawCheckerboard(resultCtx, resultCanvas.width, resultCanvas.height);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

/**
 * Draw a checkerboard pattern for transparency visualization
 */
function drawCheckerboard(ctx, width, height) {
  const tileSize = 10;
  for (let y = 0; y < height; y += tileSize) {
    for (let x = 0; x < width; x += tileSize) {
      ctx.fillStyle = ((x / tileSize + y / tileSize) % 2 === 0) ? '#ffffff' : '#cccccc';
      ctx.fillRect(x, y, tileSize, tileSize);
    }
  }
}

/**
 * Set up process button
 */
function setupProcessButton() {
  processBtn.addEventListener('click', async () => {
    if (!currentFile) {
      showError('Please select an image first');
      return;
    }

    processBtn.disabled = true;
    processBtn.textContent = 'Processing...';

    try {
      const formData = new FormData();
      formData.append('image', currentFile);

      const response = await fetch('/api/remove-background', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Processing failed');
      }

      // Get the result blob
      currentResultBlob = await response.blob();
      currentResultUrl = URL.createObjectURL(currentResultBlob);

      // Display results
      displayResult();
      resultsArea.style.display = 'block';

    } catch (error) {
      showError(error.message || 'An error occurred during processing');
    } finally {
      processBtn.disabled = false;
      processBtn.textContent = 'Process Image';
    }
  });
}

/**
 * Display the processed result
 */
function displayResult() {
  if (!currentResultUrl) return;

  const img = new Image();
  img.onload = () => {
    // Draw on result canvas with checkerboard background
    const ctx = resultCanvas.getContext('2d');
    ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    drawCheckerboard(ctx, resultCanvas.width, resultCanvas.height);
    ctx.drawImage(img, 0, 0);

    // Draw on output canvas (clean result)
    const outputCtx = outputCanvas.getContext('2d');
    outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    outputCtx.drawImage(img, 0, 0);

    // Update output info
    outputSize.textContent = `${img.width} x ${img.height} px`;

    // Enable download
    downloadBtn.disabled = false;
  };
  img.src = currentResultUrl;
}

/**
 * Set up comparison slider
 */
function setupComparisonSlider() {
  comparisonSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    updateComparisonView(value);
  });
}

function updateComparisonView(value) {
  const leftCanvas = originalCanvas;
  const rightCanvas = resultCanvas;

  // Create a combined view
  const combinedCanvas = document.createElement('canvas');
  combinedCanvas.width = leftCanvas.width;
  combinedCanvas.height = leftCanvas.height;
  const ctx = combinedCanvas.getContext('2d');

  // Draw the left side (original)
  ctx.drawImage(leftCanvas, 0, 0);

  // Clip and draw the right side (result)
  const clipWidth = (value / 100) * leftCanvas.width;
  ctx.save();
  ctx.beginPath();
  ctx.rect(clipWidth, 0, leftCanvas.width - clipWidth, leftCanvas.height);
  ctx.clip();
  ctx.drawImage(rightCanvas, 0, 0);
  ctx.restore();

  // Draw slider line
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(clipWidth, 0);
  ctx.lineTo(clipWidth, leftCanvas.height);
  ctx.stroke();

  // Draw slider handle
  ctx.fillStyle = '#3b82f6';
  ctx.beginPath();
  ctx.arc(clipWidth, leftCanvas.height / 2, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.moveTo(clipWidth - 4, leftCanvas.height / 2 - 6);
  ctx.lineTo(clipWidth + 4, leftCanvas.height / 2);
  ctx.lineTo(clipWidth - 4, leftCanvas.height / 2 + 6);
  ctx.fill();

  // Update result canvas with combined view
  const resultCtx = resultCanvas.getContext('2d');
  resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
  resultCtx.drawImage(combinedCanvas, 0, 0);
}

/**
 * Set up download button
 */
function setupDownloadButton() {
  downloadBtn.addEventListener('click', () => {
    if (!currentResultBlob) return;

    const url = URL.createObjectURL(currentResultBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `background-removed-${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

/**
 * Show error message
 */
function showError(message) {
  alert(`Error: ${message}`);
}

// Expose for testing
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { init, handleFiles, displayResult };
}