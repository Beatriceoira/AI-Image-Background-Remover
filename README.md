# MatteForge - AI Background Remover

An AI-powered image background remover and foreground extraction system using U²-Net segmentation.

## Features

- **Real AI Segmentation**: Uses U²-Net model for genuine foreground detection
- **Transparent PNG Output**: Preserves soft alpha edges for hair, fur, and fine details
- **Background Replacement**: Support for transparent, solid color, gradient, and image backgrounds
- **Configurable Refinement**: Edge smoothing, hole filling, median filtering, color decontamination
- **API-First Design**: FastAPI with Swagger docs at `/docs`
- **Docker Support**: CPU and GPU variants available
- **Comprehensive Testing**: 76+ tests covering validation, segmentation, matting, and API
- **Benchmarking**: Measure preprocessing, inference, matting, and encoding times

## Architecture

```
User
 │
 ▼
Web Frontend (HTML/CSS/Vanilla JS)
 │
 │ POST image
 ▼
FastAPI Backend
 │
 ▼
Image Validation (MIME, magic bytes, size, dimensions)
 │
 ▼
Image Preprocessing (EXIF, RGB, resize, normalize)
 │
 ▼
AI Segmentation Model (U²-Net)
 │
 ▼
Foreground Mask (soft probability map)
 │
 ▼
Mask Refinement (blur, median, hole fill, edge smooth)
 │
 ▼
Alpha Matte Generation (RGBA compositing)
 │
 ▼
PNG Output
 ├── Transparent background
 ├── Solid-color background
 ├── Gradient background
 └── Custom background image
 │
 ▼
Frontend Preview & Download
```

## Tech Stack

### Frontend
- HTML5, CSS3, Vanilla JavaScript
- Fetch API, Drag-and-drop upload
- Responsive design

### Backend
- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic

### Computer Vision
- PyTorch
- Pillow (PIL)
- NumPy
- SciPy (for morphological operations)

### Model
- U²-Net (U²Net) - Salient object detection model
- Extensible architecture for future models (MODNet, SAM, SAM2)

## Project Structure

```
matteforge/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application with lifespan
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes_health.py       # Health endpoint
│   │   │   └── routes_remove.py       # Background removal endpoint
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py              # Configuration from env vars
│   │   │   └── logging_config.py      # Logging setup
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # SegmentationModel interface
│   │   │   ├── u2net.py               # U²-Net architecture + wrapper
│   │   │   └── weights/
│   │   │       └── u2net.pth          # Model weights (gitignored)
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── inference.py           # Model singleton with warmup
│   │   │   ├── preprocessing.py       # Image preprocessing pipeline
│   │   │   ├── segmentation.py        # Mask generation
│   │   │   ├── matting.py             # Alpha compositing
│   │   │   ├── postprocessing.py      # Mask refinement
│   │   │   └── image_service.py       # Image I/O utilities
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── image_utils.py         # Encoding/decoding helpers
│   │       └── validation.py          # Input validation
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_api.py                # API integration tests
│   │   ├── test_health.py             # Health endpoint tests
│   │   ├── test_upload.py             # Upload validation tests
│   │   ├── test_segmentation.py       # Segmentation service tests
│   │   └── test_matting.py            # Matting service tests
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── index.html                     # Main HTML page
│   ├── style.css                      # Stylesheet
│   └── app.js                         # Frontend logic
│
├── scripts/
│   ├── download_model.py              # Model download script
│   └── benchmark.py                   # Performance benchmarking
│
├── uploads/                           # Temporary uploads (gitignored)
├── outputs/                           # Processed outputs (gitignored)
├── models/                            # Model weights directory (gitignored)
│
├── Dockerfile                         # CPU Docker image
├── docker-compose.yml                 # Docker Compose config
├── requirements.txt                   # CPU requirements
├── requirements-gpu.txt               # GPU requirements
├── .env.example                       # Environment variables example
├── .gitignore                         # Git ignore rules
├── README.md                          # This file
└── LICENSE                            # MIT License
```

## Requirements

- Python 3.11 or 3.12
- PyTorch (CPU or GPU)
- FastAPI
- Pillow
- NumPy
- SciPy
- Uvicorn

## Installation

### Ubuntu/Linux

```bash
# Clone or navigate to the project
cd matteforge

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (CPU version)
pip install -r requirements.txt

# Or install GPU version (requires CUDA)
# pip install -r requirements-gpu.txt
```

### Download Model Weights

The U²-Net model weights are available in `backend/app/models/weights/u2net.pth`.

If you need to download them:

```bash
python scripts/download_model.py
```

**Model Source**: [U²-Net GitHub](https://github.com/xuebinqin/U-2-Net)
**License**: BSD 3-Clause License

## Running the Application

### Development Mode

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:
- **API**: http://localhost:8000
- **Frontend**: http://localhost:8000 (serves index.html)
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Production Mode

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

**CPU版本（推荐，兼容性好）：**
```bash
docker compose up --build
```

**GPU版本（需要NVIDIA Container Toolkit）：**
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

**手动构建：**
```bash
# CPU版本
docker build -f Dockerfile.cpu -t matteforge:cpu .

# GPU版本
docker build -f Dockerfile.gpu -t matteforge:gpu .
```

## API Endpoints

### GET /health

Returns the health status, model name, and device.

**Response:**
```json
{
  "status": "ok",
  "model": "u2net",
  "device": "cuda",
  "model_loaded": true
}
```

### POST /api/remove-background

Accepts an image file and returns a processed image.

**Parameters (Query):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `background_mode` | string | `transparent` | `transparent`, `solid`, `gradient`, `image` |
| `background_color` | string | `null` | Hex color for solid background (e.g., `#FFFFFF`) |
| `refine_edges` | bool | `false` | Apply edge refinement pipeline |
| `blur_radius` | int | `0` | Gaussian blur radius (0-10) |
| `median_filter` | int | `0` | Median filter radius (0-10) |
| `hole_fill` | bool | `true` | Fill small holes in mask |
| `edge_smooth` | bool | `true` | Morphological edge smoothing |
| `color_decontamination` | bool | `false` | Remove background color bleed |
| `output_format` | string | `PNG` | `PNG`, `JPEG`, `WEBP` |

**Parameters (Form Data):**
| Parameter | Required | Description |
|-----------|----------|-------------|
| `image` | Yes | Foreground image file |
| `background_image` | No | Background image file (required for `image` mode) |

**Example using curl:**

```bash
curl \
  -X POST \
  -F "image=@photo.jpg" \
  http://localhost:8000/api/remove-background \
  --output result.png
```

**Example with background replacement:**

```bash
curl \
  -X POST \
  -F "image=@photo.jpg" \
  -F "background_image=@bg.jpg" \
  "http://localhost:8000/api/remove-background?background_mode=image&output_format=PNG" \
  --output result.png
```

**Example with solid color background:**

```bash
curl \
  -X POST \
  -F "image=@photo.jpg" \
  "http://localhost:8000/api/remove-background?background_mode=solid&background_color=#FF0000" \
  --output result.png
```

### GET /api/remove-background/info

Returns information about the endpoint parameters and capabilities.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `u2net` | Name of the model to use |
| `MODEL_PATH` | Auto-detected | Path to model weights file |
| `DEVICE` | `auto` | Device: auto, cpu, cuda |
| `ENABLE_GPU` | `True` | Enable GPU acceleration |
| `MAX_UPLOAD_SIZE_MB` | `15` | Maximum upload size in MB |
| `MAX_IMAGE_WIDTH` | `6000` | Maximum image width |
| `MAX_IMAGE_HEIGHT` | `6000` | Maximum image height |
| `OUTPUT_FORMAT` | `PNG` | Default output image format |
| `LOG_LEVEL` | `INFO` | Logging level |

Create a `.env` file from the example:

```bash
cp .env.example .env
```

## Project Structure

```
matteforge/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application
│   │   ├── api/
│   │   │   ├── routes_health.py    # Health endpoint
│   │   │   └── routes_remove.py    # Background removal endpoint
│   │   ├── core/
│   │   │   ├── config.py           # Configuration
│   │   │   ├── logging_config.py   # Logging setup
│   │   │   ├── models/
│   │   │   │   ├── base.py         # Model interface
│   │   │   │   ├── u2net.py        # U²-Net implementation
│   │   │   │   └── weights/        # Model weights (gitignored)
│   │   │   └── services/
│   │   │       ├── inference.py    # Model loading & inference
│   │   │       ├── preprocessing.py # Image preprocessing
│   │   │       ├── segmentation.py # Mask generation
│   │   │       ├── matting.py      # Alpha compositing
│   │   │       └── postprocessing.py # Mask refinement
│   │   ├── utils/
│   │   │   ├── image_utils.py      # Image encoding/decoding
│   │   │   └── validation.py       # Input validation
│   │   └── services/
│   ├── tests/                      # Test files
│   └── requirements.txt
├── frontend/                       # Web interface
├── scripts/
│   ├── download_model.py           # Model download script
│   └── benchmark.py                # Benchmarking script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-gpu.txt
├── .env.example
├── .gitignore
└── README.md
```

## Testing

Run all tests:
```bash
source .venv/bin/activate
python -m pytest backend/tests/ -v
```

Run specific test files:
```bash
python -m pytest backend/tests/test_health.py -v
python -m pytest backend/tests/test_upload.py -v
python -m pytest backend/tests/test_segmentation.py -v
python -m pytest backend/tests/test_matting.py -v
python -m pytest backend/tests/test_api.py -v
```

Test coverage includes:
- Health endpoint functionality
- Image upload validation (MIME, magic bytes, size, dimensions)
- Segmentation mask generation and dimensions
- Matting with different background modes
- Full API integration tests

## Benchmarking

Measure performance:
```bash
python scripts/benchmark.py
```

Options:
```bash
python scripts/benchmark.py --image photo.jpg --iterations 5 --device cuda
```

Example output:
```
============================================================
  Results Summary
============================================================

  Image:       512x512
  Device:      cuda
  Model:       U²-Net (44,009,869 params)

  Preprocess:      1.85 ms
  Inference:      11.19 ms
  Segmentation:   37.06 ms
  Matting:         1.42 ms
  Encoding:       23.39 ms
  ──────────────────────────
  Total:          63.72 ms
  FPS:            15.69
  GPU memory allocated: 174.7 MB
  GPU memory reserved: 596.0 MB
```

## Security

- **File Validation**: MIME type checking, magic-byte signature verification
- **Size Limits**: Configurable maximum upload size (default 15MB)
- **Dimension Limits**: Maximum image dimensions (default 6000x6000)
- **Decompression Bomb Protection**: Limits on image pixel count
- **Safe Filenames**: Server-side unique filename generation
- **Temporary Processing**: Uploaded images are processed in memory and not stored permanently
- **No Arbitrary Execution**: Uploaded files are never executed or interpreted

## Privacy

- Images are processed locally on your server
- No data is sent to external services
- Uploaded files are not retained after processing
- No analytics, tracking, or user profiling

## Model Attribution

This project uses the U²-Net model:

**Paper**: Li, Zhibin et al. "U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection." *Pattern Recognition* 106 (2020): 107402.

**GitHub**: https://github.com/xuebinqin/U-2-Net

The model weights are licensed under the BSD 3-Clause License.

## GPU Installation

For NVIDIA GPU acceleration:

1. Ensure CUDA is installed (check with `nvidia-smi`)
2. Install PyTorch with CUDA support:
   ```bash
   pip install -r requirements-gpu.txt
   ```
3. Set environment variable:
   ```bash
   export ENABLE_GPU=True
   ```

**Note**: This project uses default PyTorch wheels that auto-detect CUDA. Compatible with CUDA 11.8+ and CUDA 12.x+. Check compatibility at https://pytorch.org/.

## Docker Build Fixes

The original Dockerfile used CUDA 11.8 specific PyTorch wheels which are incompatible with CUDA 13.x. Fixed by:

1. **Updated `requirements-gpu.txt`**: Removed hardcoded CUDA 11.8 versions, using default PyTorch wheels
2. **Created `Dockerfile.cpu`**: Clean CPU-only image based on python:3.11-slim
3. **Created `Dockerfile.gpu`**: GPU image based on nvidia/cuda:13.2.0-cudnn-runtime
4. **Updated `docker-compose.yml`**: Now defaults to CPU mode
5. **Created `docker-compose.gpu.yml`**: Optional GPU override file

To run with GPU in Docker:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory errors, reduce the image size or use CPU:
```bash
export ENABLE_GPU=False
uvicorn backend.app.main:app --reload
```

### Model Not Found

If the model weights file is missing:
```bash
python scripts/download_model.py
```

### Port Already in Use

```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process or use a different port
uvicorn backend.app.main:app --reload --port 8080
```

### Docker GPU Support

For GPU support in Docker:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Ensure you have the NVIDIA Container Toolkit installed.

## Future Improvements

- [ ] Add MODNet and SAM model support
- [ ] Implement WebSocket progress updates
- [ ] Add batch processing with ZIP download
- [ ] Implement background generation
- [ ] Add AI relighting
- [ ] Shadow generation
- [ ] Image upscaling
- [ ] Object removal

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- U²-Net authors for the excellent segmentation model
- FastAPI team for the amazing web framework
- PyTorch team for the deep learning framework
- All contributors to the open-source computer vision community
