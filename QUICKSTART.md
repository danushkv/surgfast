# Quick Start Guide

Get the Surgical Tool Composite GUI running in 10 minutes!

## Prerequisites Checklist

Before starting, make sure you have:

- [ ] Python 3.8+ installed
- [ ] pip installed
- [ ] NVIDIA GPU with CUDA support
- [ ] At least 12GB GPU memory
- [ ] Base surgical images directory
- [ ] Tool regions directory (with proper structure)

## Installation (5 minutes)

### 1. Set up environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure paths

```bash
# Copy and edit config
cp config.yaml.example config.yaml
nano config.yaml  # Edit with your actual paths
```

**Minimum required edits in config.yaml**:

```yaml
data:
  no_tools_images_dir: "/your/path/to/base/images"
  tool_regions_dir: "/your/path/to/tool_regions"
  output_dir: "/your/output/directory"

models:
  huggingface_cache_dir: "/path/to/hf/cache"
```

## Running (2 minutes)

### Start the server

```bash
python run_server.py
```

You'll see:
```
 * Running on http://0.0.0.0:5000
 * Press CTRL+C to quit
```

### Open in browser

Visit: **http://localhost:5000**

## Using the GUI (3 minutes)

1. **Select Mode**: Learning (easy) or Generate (advanced)
2. **View Images**: See 4-6 base surgical images
3. **Add Tool**: Pick a tool and position
4. **Process**: Click "Process" to generate composite
5. **Download**: Click download button for full resolution
6. **Refresh**: Load new base images

## Typical Workflow

```
1. Load base images (automatic)
   ↓
2. Select a tool (e.g., grasper)
   ↓
3. Choose position/rotation (auto or manual)
   ↓
4. Click "Process" (2-5 seconds)
   ↓
5. View result with bounding box
   ↓
6. Download or refresh for new images
```

## Common Settings

### For Speed (1-2 sec per image)

Edit config.yaml:
```yaml
processing:
  num_inference_steps: 1

gpu:
  dtype: "float16"
```

### For Quality (5-10 sec per image)

Edit config.yaml:
```yaml
processing:
  num_inference_steps: 10
  guidance_strength: 7.5
```

### For Memory Issues

Edit config.yaml:
```yaml
features:
  use_medgemma: false       # Disable descriptions

gpu:
  dtype: "float16"          # Use half precision
  optimize_memory: true     # Enable optimizations

processing:
  num_inference_steps: 1    # Minimal refinement
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'transformers'"

```bash
pip install --upgrade transformers
```

### "CUDA out of memory"

In config.yaml, change:
```yaml
gpu:
  dtype: "float16"
```

### "No such file or directory" error

Check config.yaml paths:
- Use absolute paths (start with `/`)
- No `~` (use full path instead)
- Paths must exist and be readable

### Port 5000 already in use

In config.yaml, change:
```yaml
server:
  port: 5001  # Use different port
```

## Data Structure Required

Your directory structure must be:

```
base_images/
├── image_001.png
├── image_002.png
├── image_003.png

tool_regions/
├── grasper/
│   ├── images/
│   │   ├── tool_001.png
│   │   └── tool_002.png
│   └── masks/
│       ├── tool_001.png
│       └── tool_002.png
└── bipolar/
    ├── images/
    └── masks/
```

Each tool needs:
- RGB images with white background
- Matching grayscale masks (white = tool region)

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [SETUP.md](SETUP.md) for advanced configuration
- View [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) for changes made

## Getting Help

1. Check [SETUP.md](SETUP.md) troubleshooting section
2. Review config.yaml comments
3. Check terminal output for error messages
4. Verify GPU is available: `nvidia-smi`

## Key Files

| File | Purpose |
|------|---------|
| `run_server.py` | Main Flask application |
| `pipeline_stages.py` | Image processing pipeline |
| `config.yaml` | Your configuration (create from template) |
| `README.md` | Full documentation |
| `SETUP.md` | Detailed setup guide |

---

**That's it!** You should now have:
- ✓ Dependencies installed
- ✓ Configuration set up
- ✓ Server running
- ✓ Web interface accessible

Enjoy compositing surgical tools! 🔧

For issues or questions, refer to the full documentation in [README.md](README.md).
