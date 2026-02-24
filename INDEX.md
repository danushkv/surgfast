# Surgical Tool Composite GUI - File Guide

This directory contains all the code needed to run an interactive web-based system for generating synthetic surgical training images.

## 📁 Directory Contents

### 🚀 Getting Started (Start Here!)

1. **[QUICKSTART.md](QUICKSTART.md)** ⚡ **→ START HERE**
   - 10-minute quick start guide
   - Minimal setup instructions
   - Common configuration examples
   - Perfect for getting running fast

2. **[SETUP.md](SETUP.md)** 📋
   - Detailed step-by-step installation
   - Data preparation instructions
   - Comprehensive troubleshooting
   - Performance optimization guide
   - GPU/CUDA setup help

### 📖 Documentation

3. **[README.md](README.md)** 📚
   - Complete feature documentation
   - System requirements
   - Data structure requirements
   - Usage guide with examples
   - Configuration reference
   - Architecture explanation
   - Advanced usage

4. **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** ✨
   - Summary of code improvements
   - Hardcoded paths removed
   - Documentation added
   - Code quality improvements
   - Testing checklist

### ⚙️ Configuration

5. **[config.yaml](config.yaml)** 🔧
   - Configuration template
   - All customizable settings
   - Well-commented sections
   - Default values
   - **COPY THIS AND EDIT WITH YOUR PATHS**

### 💻 Application Code

6. **[run_server.py](run_server.py)** 🌐
   - Flask web server
   - API endpoints
   - GPU memory management
   - Interactive GUI backend

7. **[pipeline_stages.py](pipeline_stages.py)** 🔄
   - Image processing pipeline
   - Tool compositing
   - Stable Diffusion integration
   - MedGemma integration
   - Utility functions

### 📦 Dependencies

8. **[requirements.txt](requirements.txt)** 📝
   - Python package dependencies
   - Pinned versions for reproducibility
   - Ready for `pip install -r requirements.txt`

### 🙈 Git Configuration

9. **[.gitignore](.gitignore)** 🚫
   - GitHub upload configuration
   - Prevents committing large files
   - Protects local configurations

---

## 🚀 Quick Start Workflow

### First Time Users → Follow This Path:

1. Read [QUICKSTART.md](QUICKSTART.md) (5 min read)
2. Run the installation steps (5 min setup)
3. Configure [config.yaml](config.yaml) with your paths
4. Start the server: `python run_server.py`
5. Open browser: http://localhost:5000

### Encountering Issues? → Try This:

1. Check [SETUP.md](SETUP.md) Troubleshooting section
2. Review [config.yaml](config.yaml) comments
3. Read relevant section in [README.md](README.md)
4. Check error message in terminal

### Want to Understand Everything? → Read In Order:

1. [README.md](README.md) - Overview & features
2. [SETUP.md](SETUP.md) - Detailed setup
3. Code files with docstrings

---

## 📊 File Purposes at a Glance

| File | Type | Purpose | Read Time |
|------|------|---------|-----------|
| QUICKSTART.md | Guide | Fast setup | 5 min |
| SETUP.md | Guide | Detailed setup | 15 min |
| README.md | Docs | Full documentation | 20 min |
| config.yaml | Config | YOUR SETTINGS | varies |
| run_server.py | Code | Flask server | 10 min |
| pipeline_stages.py | Code | Processing pipeline | 15 min |
| requirements.txt | Config | Dependencies | 1 min |
| .gitignore | Config | Git settings | 1 min |

---

## 🛠️ Setup Checklist

- [ ] Install Python 3.8+
- [ ] Create virtual environment: `python3 -m venv venv`
- [ ] Activate: `source venv/bin/activate`
- [ ] Install packages: `pip install -r requirements.txt`
- [ ] Copy config: `cp config.yaml.example config.yaml`
- [ ] Edit config.yaml with your paths
- [ ] Verify GPU: `nvidia-smi`
- [ ] Start server: `python run_server.py`
- [ ] Open browser: http://localhost:5000

---

## 🎯 Key Concepts

### Configuration-Driven
All paths and settings are in `config.yaml` - no hardcoded paths in code!

### Modular Design
- `run_server.py`: Web interface
- `pipeline_stages.py`: Processing logic
- Separate concerns for easy maintenance

### Features
- Interactive GUI for tool placement
- Stable Diffusion image refinement
- MedGemma image descriptions (optional)
- Bounding box generation
- Full image downloads

### GPU Required
- 12GB+ VRAM minimum
- 24GB+ recommended (especially with MedGemma)
- NVIDIA CUDA required

---

## 📋 System Requirements

```
Python:       3.8+
GPU Memory:   12GB minimum (24GB+ recommended)
Disk Space:   30GB+ (for models and cache)
OS:           Linux (Ubuntu 20.04+)
RAM:          16GB+
Internet:     For downloading models
```

---

## 🔑 Key Files to Edit

### 1. config.yaml (REQUIRED)
```yaml
data:
  no_tools_images_dir: "/YOUR/PATH/TO/IMAGES"
  tool_regions_dir: "/YOUR/PATH/TO/TOOLS"
  output_dir: "./output"
```

### 2. (Optional) run_server.py
Only if you want to customize:
- Server host/port
- Default routes
- API endpoints

### 3. (Optional) pipeline_stages.py
Only if you want to customize:
- Processing parameters
- Model architectures
- Pipeline stages

---

## 📞 Support Resources

### Troubleshooting
- [SETUP.md - Troubleshooting](SETUP.md#common-issues-and-solutions)
- [README.md - Troubleshooting](README.md#troubleshooting)

### Data Preparation
- [SETUP.md - Data Setup](SETUP.md#step-5-prepare-your-data)
- [README.md - Data Structure](README.md#data-structure)

### Configuration Help
- [SETUP.md - Configuration](SETUP.md#configuration-deep-dive)
- [README.md - Configuration](README.md#configuration-options)
- [config.yaml](config.yaml) - Inline comments

### Performance Tuning
- [SETUP.md - Optimization](SETUP.md#performance-optimization)
- [README.md - Performance](README.md#performance-notes)

---

## ✨ What's Included

### Refactored Code ✓
- ✅ No hardcoded paths
- ✅ Comprehensive docstrings
- ✅ Better error handling
- ✅ Clean, readable code
- ✅ Configuration-driven

### Complete Documentation ✓
- ✅ Quick start guide
- ✅ Detailed setup instructions
- ✅ Full API documentation
- ✅ Troubleshooting guide
- ✅ Performance optimization guide
- ✅ Code comments throughout

### Ready for GitHub ✓
- ✅ Appropriate .gitignore
- ✅ No sensitive data
- ✅ Reproducible builds
- ✅ Clear instructions
- ✅ MIT-compatible code

---

## 🎓 Learning Path

### Beginner
1. Read QUICKSTART.md
2. Follow installation
3. Use default config settings
4. Experiment with GUI

### Intermediate
1. Read SETUP.md
2. Understand config options
3. Optimize for your GPU
4. Customize processing parameters

### Advanced
1. Read all documentation
2. Modify pipeline_stages.py
3. Add custom features
4. Integrate with other systems

---

## 📝 Notes

- **config.yaml** is in the template. Create your own copy with your actual paths.
- **First run** will download models (~20GB), may take 30+ minutes
- **GPU memory** is critical - disable features if running out of memory
- **Paths** must be absolute (start with `/`) and must exist
- **Data structure** must match tool_regions format exactly

---

## 🚀 Ready to Start?

### For Quick Start:
👉 Read [QUICKSTART.md](QUICKSTART.md)

### For Detailed Setup:
👉 Read [SETUP.md](SETUP.md)

### For Full Documentation:
👉 Read [README.md](README.md)

### For Code Understanding:
👉 Read [run_server.py](run_server.py) and [pipeline_stages.py](pipeline_stages.py)

---

**Good luck! 🎯**

Questions? Check the relevant documentation file above.
