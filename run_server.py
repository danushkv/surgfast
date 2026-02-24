"""
Flask-based GUI for surgical tool composite image generation.

This module provides an interactive web interface for:
1. Selecting base surgical images
2. Adding tool regions to images
3. Rotating and positioning tools
4. Refining results through Stable Diffusion
5. Viewing and downloading processed images

Configuration is read from config.yaml. Update paths there before running.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
from PIL import Image
import os
import random
import base64
from io import BytesIO
import torch
from diffusers import LCMScheduler, UNet2DConditionModel, StableDiffusionImg2ImgPipeline
from flask import Flask, render_template, jsonify, request, send_file
import threading
import logging
import yaml

# Import from pipeline_stages module
from pipeline_stages import (
    ToolCompositor, crop_image_sides, rotate_image, 
    process_composite_stage, initialize_medgemma_pipeline,
    run_image_through_medgemma, load_config
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
compositor = None
base_images = []
sd_pipeline = None
processing_lock = threading.Lock()
processing_results = {}
session_description = None
description_shown = False
config = None


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for embedding in HTML."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def init_compositor():
    """Initialize the ToolCompositor and load base images."""
    global compositor, base_images, sd_pipeline, session_description, description_shown, config
    
    # Load configuration
    config = load_config()
    
    # Set max content length from config
    app.config['MAX_CONTENT_LENGTH'] = config['server']['max_content_length_mb'] * 1024 * 1024
    
    logger.info("Initializing ToolCompositor...")
    compositor = ToolCompositor(
        no_tools_dir=config['data']['no_tools_images_dir'],
        tool_regions_dir=config['data']['tool_regions_dir'],
        output_dir=config['data']['output_dir']
    )
    
    # Load 4 random base images
    logger.info("Loading base images...")
    base_images = random.sample(
        compositor.no_tools_images,
        min(4, len(compositor.no_tools_images))
    )
    
    logger.info(f"Loaded {len(base_images)} base images")
    
    # Initialize MedGemma if enabled
    if config['features']['use_medgemma']:
        logger.info("Loading MedGemma pipeline...")
        try:
            medgemma_pipe, medgemma_processor = initialize_medgemma_pipeline(
                cache_dir=config['models']['huggingface_cache_dir']
            )
            logger.info("✓ MedGemma pipeline loaded")
            
            # Generate description for first base image
            logger.info("Generating description for first base image...")
            session_description = None
            description_shown = False
            
            try:
                first_img_path = base_images[0]
                base_img = Image.open(str(first_img_path)).convert('RGB')
                base_img_resized = base_img.resize((512, 512), Image.Resampling.LANCZOS)
                
                logger.info(f"  Running first base image through MedGemma...")
                description, _ = run_image_through_medgemma(
                    image=base_img_resized,
                    pipeline=medgemma_pipe,
                    processor=medgemma_processor,
                    tool_name=""
                )
                session_description = description
                logger.info(f"  ✓ Session description generated: {len(description)} chars")
            except Exception as e:
                logger.error(f"  ✗ Error generating description: {str(e)}")
                session_description = "Description generation failed."
            
            # Clean up MedGemma to free GPU memory
            logger.info("Cleaning up MedGemma pipeline...")
            del medgemma_pipe
            del medgemma_processor
            torch.cuda.empty_cache()
            logger.info("✓ MedGemma pipeline removed, GPU memory freed")
            
        except Exception as e:
            logger.error(f"Failed to initialize MedGemma: {str(e)}")
            session_description = "Description not available."
            description_shown = False
    
    # Load SD pipeline if enabled
    if config['features']['use_diffusion']:
        logger.info("Loading Stable Diffusion pipeline...")
        sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            config['models']['sd_base_model'],
            safety_checker=None,
            cache_dir=config['models']['huggingface_cache_dir'],
            torch_dtype=torch.float16 if config['gpu']['device'] == 'cuda' else torch.float32
        )
        
        # Load and set the distilled UNet
        unet = UNet2DConditionModel.from_pretrained(
            config['models']['sd_unet_checkpoint'],
            subfolder="unet",
            torch_dtype=torch.float16 if config['gpu']['device'] == 'cuda' else torch.float32
        )
        sd_pipeline.unet = unet
        sd_pipeline = sd_pipeline.to(config['gpu']['device'])
        sd_pipeline.scheduler = LCMScheduler.from_config(sd_pipeline.scheduler.config)
        logger.info("✓ SD pipeline loaded and ready")
    else:
        logger.info("Skipping Stable Diffusion pipeline (disabled in config)")


@app.route('/')
def index():
    """Render the landing page with mode selection."""
    return render_template('composite_gui_landing.html')


@app.route('/gui')
def gui():
    """Render the main GUI page with mode support."""
    if compositor is None:
        return "Initializing... Please refresh.", 503
    
    mode = request.args.get('mode', 'learning')
    if mode == 'generate':
        return render_template('composite_gui_generate.html')
    return render_template('composite_gui_simple.html')


@app.route('/api/base_images')
def get_base_images():
    """Return base images as base64 encoded strings."""
    global base_images
    
    if compositor is None:
        return jsonify({'error': 'Compositor not initialized'}), 500
    
    requested_count = int(request.args.get('count', len(base_images)))
    
    if requested_count > len(base_images):
        base_images = random.sample(
            compositor.no_tools_images,
            min(requested_count, len(compositor.no_tools_images))
        )
    
    images_data = []
    for idx, img_path in enumerate(base_images[:requested_count]):
        img = Image.open(str(img_path)).convert('RGB')
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        img_b64 = pil_to_base64(img)
        images_data.append({
            'id': idx,
            'name': img_path.stem,
            'image': img_b64,
            'path': str(img_path)
        })
    
    return jsonify({'images': images_data})


@app.route('/api/tools')
def get_tools():
    """Return available tools."""
    if compositor is None:
        return jsonify({'error': 'Compositor not initialized'}), 500
    
    available_tools = compositor.get_available_tools()
    tools_data = []
    for idx, tool_name in enumerate(available_tools):
        tools_data.append({
            'id': idx,
            'name': tool_name,
            'count': compositor.tools[tool_name]['count']
        })
    
    return jsonify({'tools': tools_data})


@app.route('/api/process', methods=['POST'])
def process_images():
    """Process selected images with tools and rotation."""
    global description_shown
    
    if compositor is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    if config['features']['use_diffusion'] and sd_pipeline is None:
        return jsonify({'error': 'SD pipeline not initialized'}), 500
    
    try:
        data = request.json
        selections = data.get('selections', {})
        mode = data.get('mode', 'learning')
        
        logger.info(f"Processing images in {mode} mode with selections: {selections}")
        
        results = []
        description_to_return = None
        
        with processing_lock:
            for img_id_str, config_data in selections.items():
                img_id = int(img_id_str)
                if img_id < len(base_images):
                    base_img_path = base_images[img_id]
                    selected_tool = config_data.get('tool')
                    
                    if mode == 'learning':
                        rotation = random.choice([0, 90, 180, 270])
                    else:
                        if config_data.get('random_rotation', False):
                            rotation = random.choice([0, 90, 180, 270])
                        else:
                            rotation = config_data.get('rotation', 0)
                    
                    if selected_tool:
                        logger.info(f"Processing image {img_id} with tool={selected_tool}, rotation={rotation}°, mode={mode}")
                        
                        if mode == 'learning':
                            draw_bbox = False
                        else:
                            draw_bbox = config_data.get('draw_bbox', True)
                        
                        result = process_composite_stage(
                            compositor,
                            base_img_path,
                            selected_tool,
                            compositor.output_dir,
                            img_id,
                            sd_pipeline if config['features']['use_diffusion'] else None,
                            None,
                            position=(0, 0),
                            rotation=rotation,
                            draw_bbox=draw_bbox,
                            config=config
                        )
                        
                        if result:
                            processed_img = result.get('processed_image')
                            if processed_img:
                                display_img = processed_img.copy()
                                display_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                                img_b64 = pil_to_base64(display_img)
                                
                                results.append({
                                    'id': img_id,
                                    'tool': selected_tool,
                                    'rotation': rotation,
                                    'image': img_b64,
                                    'bbox': result.get('bbox'),
                                    'status': 'success'
                                })
                            
                            processing_results[img_id] = result
                        
                        # Return session description only on FIRST API call
                        if mode == 'learning' and not description_shown and session_description:
                            description_to_return = session_description
                            description_shown = True
                            logger.info(f"✓ Returning session description (first and only time)")
        
        logger.info(f"✓ Processed {len(results)} images successfully in {mode} mode")
        response = {
            'success': True,
            'results': results
        }
        
        if description_to_return:
            response['description'] = description_to_return
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<int:image_id>')
def download_image(image_id: int):
    """Download full resolution processed image."""
    if image_id not in processing_results:
        return jsonify({'error': 'Image not found'}), 404
    
    result = processing_results[image_id]
    processed_img = result.get('processed_image')
    
    if processed_img:
        img_io = BytesIO()
        processed_img.save(img_io, 'PNG', quality=95)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', 
                        download_name=f"processed_{image_id}.png")
    
    return jsonify({'error': 'Image not found'}), 404


@app.route('/api/refresh_images', methods=['POST'])
def refresh_images():
    """Load a new set of base images."""
    global base_images
    
    if compositor is None:
        return jsonify({'error': 'Compositor not initialized'}), 500
    
    try:
        requested_count = int(request.args.get('count', 4))
        
        logger.info(f"Refreshing base images (count={requested_count})...")
        base_images = random.sample(
            compositor.no_tools_images,
            min(requested_count, len(compositor.no_tools_images))
        )
        
        processing_results.clear()
        
        logger.info("✓ Base images refreshed (session description unchanged)")
        return jsonify({
            'success': True,
            'message': 'Base images refreshed successfully'
        })
    except Exception as e:
        logger.error(f"Error refreshing images: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.teardown_appcontext
def cleanup(error):
    """Clean up GPU memory on app shutdown."""
    try:
        if sd_pipeline is not None:
            logger.info("Cleaning up GPU memory...")
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


if __name__ == '__main__':
    # Initialize components
    init_compositor()
    
    # Run Flask app
    if config:
        app.run(
            debug=config['server']['debug'],
            host=config['server']['host'],
            port=config['server']['port'],
            threaded=True
        )
    else:
        logger.error("Failed to load configuration. Exiting.")
