# Setup Guide

Detailed instructions for setting up the Surgical Tool Composite GUI.

## Prerequisites

Before starting, ensure you have:

- Python 3.8 or higher (`python3 --version`)
- pip package manager (`pip --version`)
- CUDA-capable GPU with drivers installed (`nvidia-smi`)
- At least 12GB of GPU memory (24GB+ recommended)
- Internet connection (for downloading models)

## Step-by-Step Installation

### 1. Create Working Directory

```bash
mkdir -p ~/projects/surgical-tools
cd ~/projects/surgical-tools
```

### 2. Clone Repository

```bash
git clone <repository-url> .
```

Or copy the files manually to your working directory.

### 3. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Verify activation (your prompt should show `(venv)` prefix).

### 4. Upgrade pip and Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This may take 5-10 minutes depending on your internet connection.

### 5. Prepare Your Data

#### Option A: Use Existing Datasets

If you already have organized data:

```bash
# Create a data directory
mkdir -p data/base_images
mkdir -p data/tool_regions

# Copy your files (adjust paths as needed)
cp -r /path/to/your/base_images/* data/base_images/
cp -r /path/to/your/tool_regions/* data/tool_regions/
```

#### Option B: Test with Sample Data

We provide sample data for testing:

```bash
python download_sample_data.py
```

This downloads a small dataset (~200MB) sufficient for testing.

### 6. Configure the System

Create your configuration file from the template:

```bash
cp config.yaml.example config.yaml
nano config.yaml  # Edit with your preferred editor
```

**Key settings to update**:

```yaml
data:
  # Point to your base surgical images
  no_tools_images_dir: "/full/path/to/base/images"
  
  # Point to your tool regions
  tool_regions_dir: "/full/path/to/tool_regions"
  
  # Output directory for generated composites
  output_dir: "/full/path/to/output"

models:
  # Keep defaults unless you have custom models
  sd_base_model: "stable-diffusion-v1-5/stable-diffusion-v1-5"
  huggingface_cache_dir: "/full/path/to/hf_cache"
  
gpu:
  device: "cuda"  # or "cpu" if no GPU available
```

**Tips**:
- Use absolute paths (starting with `/`)
- Ensure directories exist and have read permissions
- Create empty output directory: `mkdir -p /path/to/output`

### 7. Verify Installation

Test that everything is working:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from pipeline_stages import load_config; config = load_config(); print('Config loaded successfully')"
```

Both should print without errors.

### 8. Download Models (Optional but Recommended)

Pre-download models to speed up first run:

```bash
python -c "
from transformers import AutoProcessor, AutoModelForImageTextToText
import os

cache_dir = '/path/to/your/hf_cache'
os.makedirs(cache_dir, exist_ok=True)

# Download MedGemma (if needed)
print('Downloading MedGemma...')
processor = AutoProcessor.from_pretrained('google/medgemma-1.5-4b-it', cache_dir=cache_dir)
model = AutoModelForImageTextToText.from_pretrained('google/medgemma-1.5-4b-it', cache_dir=cache_dir)
print('✓ MedGemma downloaded')
"
```

## Running the Application

### Start the Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the server
python run_server.py
```

You should see:

```
 * Running on http://0.0.0.0:5000
```

### Access the Web Interface

Open your browser and go to:

```
http://localhost:5000
```

If accessing from another machine:

```
http://<your-machine-ip>:5000
```

## Configuration Deep Dive

### Processing Parameters

```yaml
processing:
  # How many base images to show initially
  default_base_images: 4
  
  # Image resolution for processing (512 is standard)
  image_resolution: 512
  
  # Guidance scale for Stable Diffusion (lower = more diverse)
  guidance_strength: 0.7
  
  # Number of inference steps (higher = better quality but slower)
  num_inference_steps: 4  # Use 1-4 for speed, 10-20 for quality
```

### Memory Optimization

If you get out-of-memory errors:

```yaml
gpu:
  optimize_memory: true      # Enable memory optimizations
  dtype: "float16"           # Use float16 instead of float32
```

Also consider:
- Disabling MedGemma: `use_medgemma: false`
- Reducing inference steps: `num_inference_steps: 1`
- Reducing image resolution: `image_resolution: 384`

### Feature Toggles

Control which components load:

```yaml
features:
  use_medgemma: false        # Disable if GPU memory < 20GB
  use_diffusion: true        # Enable for image refinement
  draw_bboxes: true          # Always enable for training data
```

## Common Issues and Solutions

### Issue: "No module named 'transformers'"

**Solution**:
```bash
source venv/bin/activate
pip install --upgrade transformers
```

### Issue: "CUDA out of memory"

**Solution** (in order of effectiveness):
1. Set `dtype: "float16"` in config
2. Set `num_inference_steps: 1` in config
3. Disable MedGemma: `use_medgemma: false`
4. Reduce batch size (only one image processed at a time currently)
5. Use a machine with more GPU memory

### Issue: "Cannot find tool_regions directory"

**Solution**:
1. Check the path in config.yaml is correct
2. Verify directory structure:
   ```bash
   ls tool_regions/
   # Should show: bipolar/ grasper/ scissors/ ...
   ```
3. Verify subdirectories exist:
   ```bash
   ls tool_regions/grasper/
   # Should show: images/ masks/
   ```

### Issue: "Model download fails / timeout"

**Solution**:
```bash
# Manually download with retry:
python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/medgemma-1.5-4b-it', 
                  cache_dir='/your/cache/path',
                  resume_download=True)
"
```

### Issue: Port 5000 Already in Use

**Solution**:
1. Change port in config.yaml:
   ```yaml
   server:
     port: 5001
   ```
2. Or kill the process using the port:
   ```bash
   lsof -i :5000  # Find process
   kill -9 <PID>  # Kill it
   ```

## Performance Optimization

### For Faster Processing

```yaml
processing:
  num_inference_steps: 1-2      # Minimal refinement
  
gpu:
  optimize_memory: true
  dtype: "float16"
```

### For Better Quality

```yaml
processing:
  num_inference_steps: 10-20    # Better image quality
  guidance_strength: 7.5        # Stronger guidance
```

### For MedGemma (Descriptions)

Requires 20GB+ GPU memory:

```yaml
features:
  use_medgemma: true

gpu:
  dtype: "float16"              # Critical for MedGemma
  optimize_memory: true
```

## Testing Your Setup

Run the test script to verify everything works:

```bash
python test_setup.py
```

Expected output:
```
✓ Config loaded
✓ ToolCompositor initialized
✓ Found X base images
✓ Found X tools
✓ SD pipeline ready
✓ Setup complete!
```

## Advanced: Using Remote Machines

To use the GUI from a different machine:

### Server Machine

```bash
# Edit config to allow remote connections
# Set in config.yaml:
# server:
#   host: "0.0.0.0"  # Accept connections from any IP
#   port: 5000

python run_server.py
```

### Client Machine

Access from browser:
```
http://<server-machine-ip>:5000
```

## Next Steps

1. **Start the server**: `python run_server.py`
2. **Open the GUI**: http://localhost:5000
3. **Select Learning Mode** to get started
4. **Read the main README.md** for usage instructions

## Getting Help

If you encounter issues:

1. Check the **Common Issues** section above
2. Review the configuration file format
3. Check logs in the terminal where you ran `python run_server.py`
4. Ensure all paths in config.yaml are absolute and correct

## System Requirements Summary

| Component | Requirement |
|-----------|-------------|
| Python | 3.8+ |
| GPU | NVIDIA with CUDA 11.8+ |
| GPU Memory | 12GB minimum, 24GB recommended |
| Disk Space | 30GB+ (models + cache) |
| OS | Linux (Ubuntu 20.04+) |
| RAM | 16GB+ |

## Troubleshooting GPU Issues

Check GPU availability:

```bash
# Show GPU info
nvidia-smi

# Test PyTorch CUDA
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU device:', torch.cuda.get_device_name(0))
"
```

If CUDA not available:
1. Verify GPU driver: `nvidia-smi` should work
2. Reinstall PyTorch with correct CUDA version from pytorch.org
3. Set `device: cpu` in config as last resort (very slow)

## Useful Commands

```bash
# Monitor GPU usage while running
nvidia-smi -l 1

# Check disk space
df -h

# Check active Python processes
ps aux | grep python

# View server logs in real-time
tail -f logs/*.log

# Test specific import
python -c "from pipeline_stages import ToolCompositor; print('OK')"
```

---

Once setup is complete, proceed to the main [README.md](README.md) for usage instructions.
