import torch 
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

cache_dir="."

sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            safety_checker=None,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16
        )

del sd_pipeline

model_id = "google/medgemma-1.5-4b-it"
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