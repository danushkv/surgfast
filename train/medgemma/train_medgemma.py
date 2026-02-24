import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from cholec_dataset import CholecMedGemmaDataset

# Check HF token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN is not set")


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str = "config_medgemma_cholec.yaml"):
    """
    Main training function for MedGemma on cholecystectomy dataset.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Extract configuration values
    dataset_config = config.get('dataset', {})
    training_config = config.get('training_args', {})
    model_config = config.get('model', {})
    
    dataset_root = dataset_config.get('dataset_root', 'dataset/cholecinstseg')
    seed = dataset_config.get('seed', 42)
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = CholecMedGemmaDataset(
        dataset_root=dataset_root,
        split_type='train',
        seed=seed,
    )
    
    print("Creating validation dataset...")
    val_dataset = CholecMedGemmaDataset(
        dataset_root=dataset_root,
        split_type='val',
        seed=seed,
    )
    
    # Save split information
    output_dir = training_config.get('output_dir', 'output/medgemma_cholec')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    split_info_path = Path(output_dir) / 'split_info.txt'
    train_dataset.processor.save_split_info(str(split_info_path))
    
    # Save config to output directory
    config_output_path = Path(output_dir) / 'config_used.yaml'
    with open(config_output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_output_path}")
    
    # Check GPU capability
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16 (e.g., H100, A100).")
    
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # Load model and processor
    model_id = model_config.get('model_id', 'google/medgemma-4b-it')
    cache_dir = model_config.get('cache_dir', 'cache')
    
    print(f"Loading model: {model_id}")
    
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
    
    model = AutoModelForImageTextToText.from_pretrained(model_id, cache_dir=cache_dir, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    
    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=model_config.get('lora_alpha', 16),
        lora_dropout=model_config.get('lora_dropout', 0.05),
        r=model_config.get('lora_r', 16),
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )
    
    def collate_fn(examples: list[dict[str, Any]]):
        """
        Collate function for DataLoader.
        
        Args:
            examples: List of examples with 'image' and 'messages' keys
            
        Returns:
            Batch with processed tensors
        """
        texts = []
        images = []
        for example in examples:
            images.append([example["image"]])
            texts.append(
                processor.apply_chat_template(
                    example["messages"], 
                    add_generation_prompt=False, 
                    tokenize=False
                ).strip()
            )
        
        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        # The labels are the input_ids, with the padding and image tokens masked in the loss computation
        labels = batch["input_ids"].clone()
        
        # Mask image tokens
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        # Mask tokens that are not used in the loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100
        
        batch["labels"] = labels
        return batch

    # Create training arguments from config
    args = SFTConfig(
        output_dir=training_config.get('output_dir', output_dir),
        num_train_epochs=training_config.get('num_train_epochs', 3),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        optim=training_config.get('optim', 'adamw_torch_fused'),
        logging_steps=training_config.get('logging_steps', 50),
        save_strategy=training_config.get('save_strategy', 'epoch'),
        eval_strategy=training_config.get('eval_strategy', 'steps'),
        eval_steps=training_config.get('eval_steps', 50),
        learning_rate=float(training_config.get('learning_rate', 2e-4)),
        bf16=training_config.get('bf16', True),
        max_grad_norm=float(training_config.get('max_grad_norm', 0.3)),
        warmup_ratio=float(training_config.get('warmup_ratio', 0.03)),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
        report_to=training_config.get('report_to', 'tensorboard'),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        run_name=training_config.get('run_name', 'medgemma_cholec_run'),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
    )
    
    print("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model()
    
    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    print("Training completed!")


if __name__ == "__main__":
    # Allow config path to be passed as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_medgemma_cholec.yaml"
    main(config_path)
