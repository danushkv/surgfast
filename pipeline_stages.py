"""
Composite surgical tool regions onto images.

This module provides functionality to:
1. Load and manage tool regions (images and masks)
2. Composite tools onto base images with flexible positioning and rotation
3. Process composites through Stable Diffusion for refinement
4. Generate image descriptions using MedGemma

Configuration is read from config.yaml. Update paths there before running.
Single-threaded implementation suitable for GPU inference.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
from PIL import Image
import os
import random
import yaml
from diffusers import LCMScheduler, UNet2DConditionModel, StableDiffusionImg2ImgPipeline
import torch
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
import time
from torch.nn.attention import SDPBackend, sdpa_kernel


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file (default 'config.yaml')
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config.yaml is not found
        ValueError: If required config keys are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please copy config.yaml to the current directory and update the paths."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required keys
    required_keys = ['data', 'models', 'server', 'gpu']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    return config


def crop_image_sides(image: Image.Image, left_pixels: int = 20, 
                     right_pixels: int = 20) -> Image.Image:
    """
    Crop specified pixels from left and right sides of image.
    
    Args:
        image: PIL Image to crop
        left_pixels: Number of pixels to crop from left (default 20)
        right_pixels: Number of pixels to crop from right (default 20)
        
    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    crop_box = (left_pixels, 0, width - right_pixels, height)
    cropped = image.crop(crop_box)
    return cropped


def rotate_image(image: Image.Image, angle: int) -> Image.Image:
    """
    Rotate an image by the specified angle (counterclockwise).
    
    Args:
        image: PIL Image to rotate
        angle: Rotation angle in degrees (0, 90, 180, 270)
        
    Returns:
        Rotated PIL Image
        
    Raises:
        ValueError: If angle is not 0, 90, 180, or 270
    """
    if angle == 0:
        return image
    elif angle in [90, 180, 270]:
        # Use white fill color to avoid black borders
        if image.mode == 'L':
            fillcolor = 255
        else:
            fillcolor = (255, 255, 255)
        return image.rotate(angle, expand=False, fillcolor=fillcolor)
    else:
        raise ValueError(f"Rotation angle must be 0, 90, 180, or 270 degrees, got {angle}")


class ToolCompositor:
    """
    Composite surgical tool regions onto images.
    
    Manages tool regions (images and masks) and provides methods to:
    - Load and cache tool samples
    - Composite tools onto base images
    - Handle tool positioning, scaling, and rotation
    - Generate bounding boxes for tool regions
    """
    
    def __init__(self, no_tools_dir: str, tool_regions_dir: str, output_dir: str):
        """
        Initialize the tool compositor.
        
        Args:
            no_tools_dir: Path to directory containing images without tools
            tool_regions_dir: Path to directory containing tool region subdirectories
                             (each tool subdirectory should have 'images' and 'masks' folders)
            output_dir: Directory to save composite images
        """
        self.no_tools_dir = Path(no_tools_dir)
        self.tool_regions_dir = Path(tool_regions_dir)
        self.output_dir = Path(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load available no-tools images
        self.no_tools_images = sorted(self.no_tools_dir.rglob('*.png'))
        print(f"Found {len(self.no_tools_images)} no-tools images")
        
        # Discover available tools
        self.tools = {}
        self._discover_tools()
        print(f"Found {len(self.tools)} tool types")
        
    def _discover_tools(self) -> None:
        """Discover all available tools and their regions."""
        for tool_folder in self.tool_regions_dir.iterdir():
            if tool_folder.is_dir() and (tool_folder / 'images').exists():
                tool_name = tool_folder.name
                images = sorted((tool_folder / 'images').glob('*.png'))
                masks = sorted((tool_folder / 'masks').glob('*.png'))
                
                if images and masks:
                    self.tools[tool_name] = {
                        'images': images,
                        'masks': masks,
                        'count': len(images)
                    }
                    print(f"  {tool_name}: {len(images)} samples")
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return sorted(self.tools.keys())
    
    def _get_bbox_from_mask(self, mask: np.ndarray, threshold: int = 127) -> Tuple[int, int, int, int]:
        """
        Extract bounding box coordinates from a mask.
        
        Args:
            mask: Binary mask as numpy array (grayscale)
            threshold: Threshold for mask detection (default 127)
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        binary_mask = (mask > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (0, 0, mask.shape[1], mask.shape[0])
        
        x_min, y_min = mask.shape[1], mask.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(mask.shape[1], x_max)
        y_max = min(mask.shape[0], y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return (0, 0, mask.shape[1], mask.shape[0])
        
        return (x_min, y_min, x_max, y_max)
    
    def draw_bbox_on_image(self, image: Image.Image, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3) -> Image.Image:
        """
        Draw a bounding box on an image.
        
        Args:
            image: PIL Image (RGB)
            bbox: Tuple of (x_min, y_min, x_max, y_max)
            color: RGB color tuple (default green)
            thickness: Line thickness in pixels (default 3)
            
        Returns:
            Image with bounding box drawn (PIL Image)
        """
        image = image.copy()
        img_array = np.array(image, dtype=np.uint8)
        
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(img_array.shape[1], int(x_max))
        y_max = min(img_array.shape[0], int(y_max))
        
        if x_max > x_min and y_max > y_min:
            cv2.rectangle(img_array, (x_min, y_min), (x_max, y_max), color, thickness)
        
        result = Image.fromarray(img_array, mode='RGB')
        return result
    
    def load_tool_sample(self, tool_name: str, sample_idx: int, crop_borders: bool = True) -> Tuple[Image.Image, Image.Image]:
        """
        Load a tool image and its mask.
        
        Args:
            tool_name: Name of the tool
            sample_idx: Index of the sample (random if None)
            crop_borders: If True, automatically crop black borders
            
        Returns:
            Tuple of (tool_image, tool_mask) as PIL Images
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool_info = self.tools[tool_name]
        
        if sample_idx is None:
            sample_idx = random.randint(0, tool_info['count'] - 1)
        elif sample_idx >= tool_info['count']:
            sample_idx = sample_idx % tool_info['count']
        
        img_path = tool_info['images'][sample_idx]
        mask_path = tool_info['masks'][sample_idx]
        
        # Load image and mask
        tool_img = cv2.imread(str(img_path))
        tool_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert to PIL Images
        tool_img = cv2.cvtColor(tool_img, cv2.COLOR_BGR2RGB)
        tool_img = Image.fromarray(tool_img)
        tool_mask = Image.fromarray(tool_mask)
        
        # Crop left and right sides
        tool_img = crop_image_sides(tool_img, left_pixels=20, right_pixels=20)
        tool_mask = crop_image_sides(tool_mask, left_pixels=20, right_pixels=20)
        
        return tool_img, tool_mask
    
    def crop_tool_to_base_dimensions(self, tool_image: Image.Image, tool_mask: Image.Image,
                                      base_image: Image.Image, method: str = 'center') -> Tuple[Image.Image, Image.Image]:
        """
        Crop tool image and mask to match base image dimensions.
        
        Args:
            tool_image: Tool image (PIL Image RGB)
            tool_mask: Tool mask (PIL Image grayscale)
            base_image: Base image to match dimensions from
            method: Cropping method - 'center' (default), 'top', 'bottom'
            
        Returns:
            Tuple of (cropped_tool_image, cropped_tool_mask)
        """
        base_w, base_h = base_image.size
        tool_w, tool_h = tool_image.size
        
        if tool_w <= base_w and tool_h <= base_h:
            return tool_image, tool_mask
        
        # Crop width if needed
        if tool_w > base_w:
            if method == 'center':
                left = (tool_w - base_w) // 2
            elif method == 'top':
                left = 0
            else:  # bottom
                left = tool_w - base_w
            
            right = left + base_w
            tool_image = tool_image.crop((left, 0, right, tool_h))
            tool_mask = tool_mask.crop((left, 0, right, tool_h))
        
        # Crop height if needed
        tool_w, tool_h = tool_image.size
        if tool_h > base_h:
            if method == 'center':
                top = (tool_h - base_h) // 2
            elif method == 'top':
                top = 0
            else:  # bottom
                top = tool_h - base_h
            
            bottom = top + base_h
            tool_image = tool_image.crop((0, top, tool_w, bottom))
            tool_mask = tool_mask.crop((0, top, tool_w, bottom))
        
        return tool_image, tool_mask
    
    def paste_tool_excluding_white(self, base_image: Image.Image, tool_image: Image.Image,
                                   position: Tuple[int, int] = (10, 10), 
                                   white_threshold: int = 254) -> Image.Image:
        """
        Paste tool image onto base image, excluding white pixels.
        
        Args:
            base_image: Base image (PIL Image RGB)
            tool_image: Tool image (PIL Image RGB) with white background
            position: (x, y) position to paste tool
            white_threshold: Threshold for white detection (default 254)
            
        Returns:
            Composite image (PIL Image)
        """
        result = base_image.copy()
        tool_array = np.array(tool_image, dtype=np.uint8)
        
        # Create mask for non-white pixels
        white_mask = (tool_array[:, :, 0] > white_threshold) & \
                     (tool_array[:, :, 1] > white_threshold) & \
                     (tool_array[:, :, 2] > white_threshold)
        
        # Convert tool to RGBA
        tool_rgba = tool_image.convert('RGBA')
        
        # Create alpha channel where white pixels are 0 (transparent)
        alpha_array = np.ones((tool_image.height, tool_image.width), dtype=np.uint8) * 255
        alpha_array[white_mask] = 0
        alpha_channel = Image.fromarray(alpha_array, mode='L')
        
        # Apply alpha to tool_rgba
        tool_rgba.putalpha(alpha_channel)
        
        # Paste tool onto result using the alpha mask
        result.paste(tool_rgba, position, tool_rgba)
        
        return result


def initialize_medgemma_pipeline(
    model_id: str = "google/medgemma-1.5-4b-it",
    cache_dir: str = None
):
    """
    Initialize the MedGemma pipeline for image description generation.
    
    Args:
        model_id: HuggingFace model ID (default google/medgemma-1.5-4b-it)
        cache_dir: Cache directory for models
        ft_checkpoint_dir: Path to fine-tuned LoRA checkpoint
        
    Returns:
        Tuple of (pipeline, processor)
    """
    print("Loading MedGemma processor...")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)

    model_kwargs = dict(
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Quantization config
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["dtype"],
        bnb_4bit_quant_storage=model_kwargs["dtype"],
    )
    
    print("Loading base MedGemma model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        **model_kwargs
    )
    
    model = base_model.eval()
    
    ft_pipe = pipeline(
        "image-text-to-text",
        model=model,
        processor=processor,
        cache_dir=cache_dir,
        device_map="auto"
    )
    
    # Configure generation settings
    ft_pipe.model.generation_config.do_sample = False
    ft_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"
    
    return ft_pipe, processor


def run_image_through_medgemma(
    image: Image.Image,
    pipeline: Any,
    processor: Any,
    tool_name: str,
    system_instruction: str = "You are a helping assistant for learning about the tools in a surgical scene."
) -> Tuple[str, Any]:
    """
    Run an image through the MedGemma pipeline to generate descriptive text.
    
    Args:
        image: PIL Image to process
        pipeline: Initialized MedGemma pipeline
        processor: Processor for the model
        tool_name: Name of the tool (for context)
        system_instruction: System instruction for the model
        
    Returns:
        Tuple of (description_text, raw_outputs)
    """
    prompt = "Describe the surgical tools:bipolar forceps, clipper, snare, scissors, grasper. Explain what a laparoscopic cholecystectomy procedure is and what each tool is used for."
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]
    
    outputs = pipeline(
        text=messages,
        max_new_tokens=1000,
        do_sample=False
    )
    
    response = outputs[0]["generated_text"][-1]["content"]
    
    return response, outputs


def process_composite_stage(
    compositor: ToolCompositor, 
    base_img_path: Path, 
    selected_tool: str, 
    output_dir: str, 
    img_idx: int, 
    sd_pipeline: Any, 
    tool_mask_data: Optional[Tuple],
    position: Tuple[int, int] = (0, 0), 
    rotation: int = 0, 
    draw_bbox: bool = True,
    config: Dict[str, Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Process a single image through the composite and SD pipeline stages.
    
    Args:
        compositor: ToolCompositor instance
        base_img_path: Path to base image
        selected_tool: Name of tool to apply
        output_dir: Output directory for saving
        img_idx: Image index
        sd_pipeline: Pre-loaded SD pipeline (can be None)
        tool_mask_data: Pre-computed tool mask data (not used, can be None)
        position: (x, y) position to place the tool
        rotation: Rotation angle in degrees (0, 90, 180, 270)
        draw_bbox: Whether to draw bounding box on the processed image
        config: Configuration dictionary
        
    Returns:
        Dictionary with processing results or None on error
    """
    try:
        # Load base image
        base_img = Image.open(str(base_img_path)).convert('RGB')
        
        # Load tool image and mask
        tool_img, tool_mask = compositor.load_tool_sample(selected_tool, sample_idx=img_idx)
        
        # Crop tool to match base image dimensions
        tool_img, tool_mask = compositor.crop_tool_to_base_dimensions(tool_img, tool_mask, base_img)
        
        # Apply rotation if specified
        if rotation != 0:
            print(f"    Applying {rotation}° rotation to tool image and mask...")
            tool_img = rotate_image(tool_img, rotation)
            tool_mask = rotate_image(tool_mask, rotation)
        
        # Create composite
        composite = compositor.paste_tool_excluding_white(base_img, tool_img, position=position)
        
        # Save composite before pipeline
        composite_filename = f"{base_img_path.stem}_{selected_tool}_composite.png"
        composite_path = os.path.join(output_dir, composite_filename)
        composite.save(composite_path)
        
        # Resize composite for processing
        resolution = 512
        if composite.mode != 'RGB':
            composite = composite.convert('RGB')
        composite = composite.resize((resolution, resolution), Image.Resampling.LANCZOS)
        
        # Create positioned mask for bbox calculation
        base_h, base_w = base_img.size[1], base_img.size[0]
        positioned_mask = Image.new('L', (base_w, base_h), 0)
        positioned_mask.paste(tool_mask, position)
        
        # Extract bounding box
        mask_array = np.array(positioned_mask, dtype=np.uint8)
        compositor_inst = ToolCompositor.__new__(ToolCompositor)
        bbox_original = compositor_inst._get_bbox_from_mask(mask_array)
        
        # Scale bbox to resolution size
        x_min, y_min, x_max, y_max = bbox_original
        scale_x = resolution / base_w
        scale_y = resolution / base_h
        
        bbox = (
            int(x_min * scale_x),
            int(y_min * scale_y),
            int(x_max * scale_x),
            int(y_max * scale_y)
        )
        
        print(f"    Position: {position}, Rotation: {rotation}°, BBox: {bbox}")
        
        # Process through SD pipeline if available
        if sd_pipeline is not None:
            pipeline_args = {
                "prompt": "cholecystectomy surgical image",
                "guidance_scale": config['processing']['guidance_strength'] if config else 4.5,
                "num_images_per_prompt": 1,
                "num_inference_steps": config['processing']['num_inference_steps'] if config else 10,
                "image": composite,
                "strength": 0.10
            }
            
            result = sd_pipeline(**pipeline_args)
            processed_image = result.images[0]
        else:
            processed_image = composite
        
        # Draw bounding box on processed image if requested
        if draw_bbox:
            processed_with_bbox = compositor_inst.draw_bbox_on_image(
                processed_image,
                bbox,
                color=(0, 255, 0),
                thickness=3
            )
            processed_filename = f"{base_img_path.stem}_{selected_tool}_processed_with_bbox.png"
        else:
            processed_with_bbox = processed_image
            processed_filename = f"{base_img_path.stem}_{selected_tool}_processed.png"
        
        # Save processed image
        processed_path = os.path.join(output_dir, processed_filename)
        processed_with_bbox.save(processed_path)
        
        print(f"  ✓ [{img_idx}] Composite & pipeline complete: {processed_filename}")
        
        return {
            'base_image': base_img_path.name,
            'tool': selected_tool,
            'composite': composite_filename,
            'processed': processed_filename,
            'bbox': bbox,
            'position': position,
            'rotation': rotation,
            'processed_image': processed_with_bbox,
            'draw_bbox': draw_bbox
        }
        
    except Exception as e:
        print(f"  ✗ [{img_idx}] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
