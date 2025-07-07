import os
import torch
import json
import safetensors.torch
from typing import Dict, Any, List, Tuple, Optional
import time
import threading
import gc
import traceback
from pathlib import Path
import glob
import shutil
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math

# Try to import LoRA-specific libraries
try:
    import diffusers
    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
    from diffusers.optimization import get_scheduler
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: diffusers not available: {e}")
    DIFFUSERS_AVAILABLE = False

try:
    import transformers
    from transformers import CLIPTextModel, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

try:
    import peft
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: peft not available: {e}")
    PEFT_AVAILABLE = False

class LoRAModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize LoRA weights
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        return x + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

# Dataset class for LoRA training
class LoRADataset(Dataset):
    def __init__(self, dataset_items: List[Dict], tokenizer: Any, resolution: int = 512):
        self.dataset_items = dataset_items
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # Image transforms
        self.transforms = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.dataset_items)
    
    def __getitem__(self, idx):
        item = self.dataset_items[idx]
        
        # Load and process image
        try:
            image = Image.open(item['image_path']).convert('RGB')
            pixel_values = self.transforms(image)
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            # Return black image on error
            pixel_values = torch.zeros(3, self.resolution, self.resolution)
        
        # Tokenize caption
        try:
            inputs = self.tokenizer(
                item['caption'],
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs.input_ids[0]
        except Exception as e:
            print(f"Error tokenizing caption: {e}")
            # Return empty token on error
            input_ids = torch.zeros(self.tokenizer.model_max_length, dtype=torch.long)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'caption': item['caption']
        }

class LoRATrainerNode:
    def __init__(self):
        self.training_thread = None
        self.stop_training = False
        self.training_status = "idle"
        self.progress = 0
        self.current_loss = 0.0
        self.best_loss = float('inf')
        self.training_history = []
        
        # ComfyUI paths
        self.comfyui_path = self._find_comfyui_path()
        
        # Try to find alternative paths if standard structure doesn't exist
        alternative_paths = self._find_alternative_paths()
        
        # Set paths with fallbacks
        self.checkpoints_path = alternative_paths.get('checkpoints', 
                                                   os.path.join(self.comfyui_path, "models", "checkpoints"))
        self.unet_path = alternative_paths.get('unet',
                                              os.path.join(self.comfyui_path, "models", "unet"))
        self.loras_path = alternative_paths.get('loras',
                                               os.path.join(self.comfyui_path, "models", "loras"))
        
        # Model cache
        self._available_models = None
        self._last_model_scan = 0
        
        # Ensure LoRA output directory exists
        self.loras_path = self._ensure_lora_output_directory()
    
    def _find_comfyui_path(self) -> str:
        """Find ComfyUI installation path by searching up the directory tree"""
        current_path = os.path.dirname(os.path.abspath(__file__))
        
        # Try to find ComfyUI by going up the directory tree
        while current_path != os.path.dirname(current_path):
            # Check for ComfyUI markers
            if (os.path.exists(os.path.join(current_path, "models", "checkpoints")) or
                os.path.exists(os.path.join(current_path, "models", "unet")) or
                os.path.exists(os.path.join(current_path, "models", "Stable-diffusion")) or
                os.path.exists(os.path.join(current_path, "main.py")) or  # ComfyUI main file
                os.path.exists(os.path.join(current_path, "execution.py"))):  # ComfyUI execution file
                return current_path
            current_path = os.path.dirname(current_path)
        
        # If we can't find ComfyUI automatically, try common locations
        common_locations = [
            os.path.join(os.path.expanduser("~"), "ComfyUI"),
            os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI"),
            os.path.join(os.path.expanduser("~"), "Desktop", "ComfyUI"),
            "C:\\ComfyUI",
            "C:\\AI\\ComfyUI",
            "C:\\tools\\ComfyUI",
            "D:\\ComfyUI",
            "D:\\AI\\ComfyUI"
        ]
        
        for location in common_locations:
            if os.path.exists(location) and os.path.exists(os.path.join(location, "models")):
                return location
        
        # Default fallback (your current setup)
        return "C:\\Users\\rosha\\Work\\apps\\ComfyUI"
    
    def _find_alternative_paths(self) -> Dict[str, str]:
        """Find alternative ComfyUI paths if standard structure doesn't exist"""
        alternative_paths = {}
        
        # Common alternative locations for ComfyUI installations
        possible_comfyui_roots = [
            self.comfyui_path,
            os.path.join(os.path.expanduser("~"), "ComfyUI"),
            os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI"),
            os.path.join(os.path.expanduser("~"), "Desktop", "ComfyUI"),
            os.path.join(os.path.expanduser("~"), "Downloads", "ComfyUI"),
            "C:\\ComfyUI",
            "C:\\AI\\ComfyUI",
            "C:\\tools\\ComfyUI",
            "C:\\stable-diffusion\\ComfyUI",
            "D:\\ComfyUI",
            "D:\\AI\\ComfyUI",
            # Portable installations
            os.path.join(os.path.expanduser("~"), "ComfyUI_windows_portable"),
            "C:\\ComfyUI_windows_portable",
            "D:\\ComfyUI_windows_portable",
            # Linux-style paths (if running on WSL)
            "/home/user/ComfyUI",
            "/opt/ComfyUI"
        ]
        
        for root in possible_comfyui_roots:
            if os.path.exists(root):
                # Check for models directory
                models_dir = os.path.join(root, "models")
                if os.path.exists(models_dir):
                    # Look for checkpoint models (various naming conventions)
                    checkpoints_candidates = [
                        os.path.join(models_dir, "checkpoints"),
                        os.path.join(models_dir, "Stable-diffusion"),  # A1111 style
                        os.path.join(models_dir, "stable-diffusion"),  # Alternative naming
                        os.path.join(models_dir, "models"),  # Generic models folder
                        os.path.join(models_dir, "sd"),  # Short form
                        os.path.join(models_dir, "base"),  # Base models
                        models_dir  # Sometimes models are directly in models folder
                    ]
                    
                    for checkpoint_path in checkpoints_candidates:
                        if os.path.exists(checkpoint_path):
                            # Check if it actually contains model files
                            if self._has_model_files(checkpoint_path):
                                alternative_paths['checkpoints'] = checkpoint_path
                                break
                    
                    # Look for unet models (ComfyUI specific)
                    unet_candidates = [
                        os.path.join(models_dir, "unet"),
                        os.path.join(models_dir, "diffusion_models"),
                        os.path.join(models_dir, "unets"),
                        os.path.join(models_dir, "unet_models")
                    ]
                    
                    for unet_path in unet_candidates:
                        if os.path.exists(unet_path):
                            if self._has_model_files(unet_path):
                                alternative_paths['unet'] = unet_path
                                break
                    
                    # Look for loras (various naming conventions)
                    lora_candidates = [
                        os.path.join(models_dir, "loras"),
                        os.path.join(models_dir, "Lora"),
                        os.path.join(models_dir, "LoRA"),
                        os.path.join(models_dir, "lora"),
                        os.path.join(models_dir, "adapters"),
                        os.path.join(models_dir, "lora_models")
                    ]
                    
                    for lora_path in lora_candidates:
                        if os.path.exists(lora_path):
                            alternative_paths['loras'] = lora_path
                            break
                    
                    # Create loras directory if it doesn't exist and we found models
                    if 'loras' not in alternative_paths and 'checkpoints' in alternative_paths:
                        default_lora_path = os.path.join(models_dir, "loras")
                        alternative_paths['loras'] = default_lora_path
                
                # If we found at least checkpoints, this is probably the right ComfyUI
                if 'checkpoints' in alternative_paths:
                    break
        
        return alternative_paths
    
    def _has_model_files(self, path: str) -> bool:
        """Check if directory contains model files (safetensors or ckpt)"""
        if not os.path.exists(path):
            return False
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.safetensors') or file.endswith('.ckpt'):
                    return True
        return False

    def _ensure_lora_output_directory(self) -> str:
        """Ensure LoRA output directory exists, create if necessary"""
        if not os.path.exists(self.loras_path):
            try:
                os.makedirs(self.loras_path, exist_ok=True)
                print(f"Created LoRA output directory: {self.loras_path}")
            except OSError as e:
                print(f"Warning: Could not create LoRA directory {self.loras_path}: {e}")
                # Fallback to user's home directory
                fallback_path = os.path.join(os.path.expanduser("~"), "ComfyUI_LoRAs")
                os.makedirs(fallback_path, exist_ok=True)
                return fallback_path
        
        return self.loras_path

    def _scan_available_models(self) -> List[Dict[str, Any]]:
        current_time = time.time()
        
        # Cache for 5 minutes
        if self._available_models and (current_time - self._last_model_scan) < 300:
            return self._available_models
        
        models = []
        
        # Scan checkpoint models
        if os.path.exists(self.checkpoints_path):
            for root, dirs, files in os.walk(self.checkpoints_path):
                for file in files:
                    if file.endswith('.safetensors') or file.endswith('.ckpt'):
                        model_path = os.path.join(root, file)
                        try:
                            model_size = os.path.getsize(model_path) / (1024**3)  # Size in GB
                        except OSError:
                            model_size = 0.0
                        
                        # Determine category based on folder structure
                        relative_path = os.path.relpath(root, self.checkpoints_path)
                        if relative_path == '.':
                            category = 'checkpoint'  # Files directly in checkpoints folder
                        else:
                            category = relative_path.replace('\\', '/').split('/')[0]  # First subfolder
                        
                        # Try to detect model type from filename
                        model_type = self._detect_model_type(file, category)
                        
                        models.append({
                            'name': file,
                            'path': model_path,
                            'type': 'checkpoint',
                            'size_gb': model_size,
                            'category': category,
                            'model_type': model_type,
                            'relative_path': relative_path
                        })
        
        # Scan UNet models
        if os.path.exists(self.unet_path):
            for root, dirs, files in os.walk(self.unet_path):
                for file in files:
                    if file.endswith('.safetensors') or file.endswith('.ckpt'):
                        model_path = os.path.join(root, file)
                        try:
                            model_size = os.path.getsize(model_path) / (1024**3)  # Size in GB
                        except OSError:
                            model_size = 0.0
                        
                        # Determine category based on folder structure
                        relative_path = os.path.relpath(root, self.unet_path)
                        if relative_path == '.':
                            category = 'unet'  # Files directly in unet folder
                        else:
                            category = relative_path.replace('\\', '/').split('/')[0]  # First subfolder
                        
                        # Try to detect model type from filename
                        model_type = self._detect_model_type(file, category)
                        
                        models.append({
                            'name': file,
                            'path': model_path,
                            'type': 'unet',
                            'size_gb': model_size,
                            'category': category,
                            'model_type': model_type,
                            'relative_path': relative_path
                        })
        
        # Sort models by type and size for better organization
        models.sort(key=lambda x: (x['model_type'], x['category'], -x['size_gb']))
        
        self._available_models = models
        self._last_model_scan = current_time
        
        return models
    
    def _get_model_dropdown_options(self) -> List[str]:
        """Get dropdown options for model selection with improved formatting"""
        models = self._scan_available_models()
        
        options = []
        for model in models:
            # Create descriptive name with model type, category, and size
            if model['model_type'] != 'Unknown':
                if model['category'] in ['checkpoint', 'unet']:
                    # No subfolder, just show type and filename
                    display_name = f"[{model['model_type']}] {model['name']} ({model['size_gb']:.1f}GB)"
                else:
                    # Has subfolder, show type, category, and filename
                    display_name = f"[{model['model_type']}] {model['category']}/{model['name']} ({model['size_gb']:.1f}GB)"
            else:
                # Unknown type, show category and filename
                if model['category'] in ['checkpoint', 'unet']:
                    display_name = f"[{model['type'].upper()}] {model['name']} ({model['size_gb']:.1f}GB)"
                else:
                    display_name = f"[{model['type'].upper()}] {model['category']}/{model['name']} ({model['size_gb']:.1f}GB)"
            
            options.append(display_name)
        
        return options if options else ["No models found"]
    
    def _detect_model_type(self, filename: str, category: str) -> str:
        """Detect model type from filename and category"""
        filename_lower = filename.lower()
        category_lower = category.lower()
        
        # Check for specific model types in filename first (most reliable)
        if any(x in filename_lower for x in ['flux', 'flux1', 'flux.1', 'flux-1']):
            return 'FLUX'
        elif any(x in filename_lower for x in ['sd3.5', 'sd35', 'sd_3.5', 'sd_35', 'sd-3.5']):
            return 'SD3.5'
        elif any(x in filename_lower for x in ['sd3.0', 'sd30', 'sd_3.0', 'sd_30', 'sd-3.0', 'sd3medium', 'sd3large']):
            return 'SD3.0'
        elif any(x in filename_lower for x in ['sdxl', 'sd_xl', 'sd-xl', 'xl_', '_xl']):
            return 'SDXL'
        elif any(x in filename_lower for x in ['sd15', 'sd_15', 'sd-15', 'sd1.5', 'sd_1.5', 'sd-1.5']):
            return 'SD1.5'
        elif any(x in filename_lower for x in ['sd2', 'sd_2', 'sd-2', 'sd2.1', 'sd_2.1', 'sd-2.1']):
            return 'SD2.x'
        
        # Check category/folder names if filename doesn't give us enough info
        if any(x in category_lower for x in ['flux', 'flux1', 'flux.1', 'flux-1']):
            return 'FLUX'
        elif any(x in category_lower for x in ['sd3.5', 'sd35', 'sd_3.5', 'sd_35', 'sd-3.5']):
            return 'SD3.5'
        elif any(x in category_lower for x in ['sd3.0', 'sd30', 'sd_3.0', 'sd_30', 'sd-3.0', 'sd3']):
            return 'SD3.0'
        elif any(x in category_lower for x in ['sdxl', 'sd_xl', 'sd-xl', 'xl', 'sdxl_1.0', 'sdxl1.0']):
            return 'SDXL'
        elif any(x in category_lower for x in ['sd15', 'sd_15', 'sd-15', 'sd1.5', 'sd_1.5', 'sd-1.5', 'sd1_5']):
            return 'SD1.5'
        elif any(x in category_lower for x in ['sd2', 'sd_2', 'sd-2', 'sd2.0', 'sd2.1']):
            return 'SD2.x'
        
        # Default fallback
        return 'Unknown'

    @classmethod
    def INPUT_TYPES(cls):
        # Create instance to get model options
        instance = cls()
        model_options = instance._get_model_dropdown_options()
        
        return {
            "required": {
                "base_model": (model_options, {
                    "default": model_options[0] if model_options else "No models found"
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-6,
                    "max": 1e-2,
                    "step": 1e-6,
                    "display": "number"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
                "num_epochs": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "rank": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "output_name": ("STRING", {
                    "default": "my_lora",
                    "multiline": False
                }),
                "dataset_path": ("STRING", {
                    "default": "path/to/training/images",
                    "multiline": False
                }),
                "trigger_word": ("STRING", {
                    "default": "my_trigger",
                    "multiline": False
                })
            },
            "optional": {
                "target_modules": ("STRING", {
                    "default": "to_k,to_q,to_v,to_out.0",
                    "multiline": False
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1
                }),
                "warmup_steps": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 1000,
                    "step": 10
                }),
                "save_every_n_epochs": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "max_train_steps": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 10000,
                    "step": 100
                }),
                "mixed_precision": (["no", "fp16", "bf16"], {
                    "default": "fp16"
                }),
                "optimizer": (["AdamW", "AdamW8bit", "Lion"], {
                    "default": "AdamW"
                }),
                "scheduler": (["cosine", "linear", "constant", "cosine_with_restarts"], {
                    "default": "cosine"
                }),
                "gradient_checkpointing": ("BOOLEAN", {
                    "default": True
                }),
                "clip_grad_norm": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "noise_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "progress", "loss", "model_path")
    OUTPUT_NODE = True
    
    FUNCTION = "train_lora"
    
    CATEGORY = "Globetrotter/Training"
    
    def train_lora(self, **kwargs):
        """Real LoRA training implementation"""
        try:
            # Check if required libraries are available
            if not DIFFUSERS_AVAILABLE:
                return ("Error: diffusers library not available. Please install with: pip install diffusers", "0%", "N/A", "")
            
            if not PEFT_AVAILABLE:
                return ("Error: peft library not available. Please install with: pip install peft", "0%", "N/A", "")
            
            # Extract parameters
            base_model = kwargs.get('base_model', '')
            learning_rate = kwargs.get('learning_rate', 1e-4)
            batch_size = kwargs.get('batch_size', 1)
            num_epochs = kwargs.get('num_epochs', 10)
            rank = kwargs.get('rank', 4)
            alpha = kwargs.get('alpha', 1.0)
            output_name = kwargs.get('output_name', 'my_lora')
            dataset_path = kwargs.get('dataset_path', '')
            trigger_word = kwargs.get('trigger_word', 'my_trigger')
            
            # Optional parameters
            target_modules = kwargs.get('target_modules', 'to_k,to_q,to_v,to_out.0')
            gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
            warmup_steps = kwargs.get('warmup_steps', 100)
            max_train_steps = kwargs.get('max_train_steps', 1000)
            mixed_precision = kwargs.get('mixed_precision', 'fp16')
            optimizer_name = kwargs.get('optimizer', 'AdamW')
            scheduler_name = kwargs.get('scheduler', 'cosine')
            gradient_checkpointing = kwargs.get('gradient_checkpointing', True)
            clip_grad_norm = kwargs.get('clip_grad_norm', 1.0)
            noise_offset = kwargs.get('noise_offset', 0.0)
            resolution = kwargs.get('resolution', 512)
            
            # Initialize training state
            self.training_status = "initializing"
            self.progress = 0
            self.current_loss = 0.0
            self.best_loss = float('inf')
            self.training_history = []
            
            # Get model information
            model_info = self._get_model_info(base_model)
            if not model_info:
                return ("Error: Selected model not found", "0%", "N/A", "")
            
            # Validate dataset path
            if not dataset_path or not os.path.exists(dataset_path):
                return ("Error: Dataset path not found", "0%", "N/A", "")
            
            # Check if model type is supported
            if model_info['model_type'] not in ['SDXL', 'SD1.5']:
                return (f"Error: Model type {model_info['model_type']} not yet supported for training", "0%", "N/A", "")
            
            print(f"Starting LoRA training for {model_info['name']} ({model_info['model_type']})")
            
            # Load dataset
            self.training_status = "loading_dataset"
            self.progress = 5
            
            dataset = self._prepare_dataset(dataset_path, trigger_word, resolution)
            if not dataset:
                return ("Error: No images found in dataset", "0%", "N/A", "")
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            print(f"Loaded dataset with {len(dataset)} images")
            
            # Load base model
            self.training_status = "loading_model"
            self.progress = 10
            
            try:
                model_components = self._load_base_model(model_info)
            except Exception as e:
                return (f"Error loading model: {str(e)}", "0%", "N/A", "")
            
            # Parse target modules
            target_modules_list = [m.strip() for m in target_modules.split(',')]
            
            # Inject LoRA layers
            self.training_status = "injecting_lora"
            self.progress = 15
            
            try:
                model_components = self._inject_lora_layers(model_components, rank, alpha, target_modules_list)
            except Exception as e:
                return (f"Error injecting LoRA: {str(e)}", "0%", "N/A", "")
            
            # Setup training
            self.training_status = "setup_training"
            self.progress = 20
            
            # Calculate training steps
            num_training_steps = min(max_train_steps, len(dataloader) * num_epochs)
            
            # Setup optimizer and scheduler
            try:
                optimizer, scheduler = self._prepare_optimizer_and_scheduler(
                    model_components, learning_rate, optimizer_name, scheduler_name,
                    num_training_steps, warmup_steps
                )
            except Exception as e:
                return (f"Error setting up training: {str(e)}", "0%", "N/A", "")
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_components['unet'].to(device)
            
            if mixed_precision != "no":
                scaler = torch.cuda.amp.GradScaler()
            else:
                scaler = None
            
            print(f"Training on {device} with {mixed_precision} precision")
            
            # Training loop
            self.training_status = "training"
            step = 0
            
            for epoch in range(num_epochs):
                if self.stop_training:
                    break
                
                epoch_loss = 0.0
                
                for batch_idx, batch in enumerate(dataloader):
                    if self.stop_training or step >= max_train_steps:
                        break
                    
                    # Move batch to device
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device)
                    
                    # Forward pass
                    try:
                        if mixed_precision != "no":
                            with torch.cuda.amp.autocast():
                                loss = self._compute_loss(model_components, batch, noise_offset)
                        else:
                            loss = self._compute_loss(model_components, batch, noise_offset)
                        
                        # Scale loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps
                        
                        # Backward pass
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        
                        # Accumulate gradients
                        if (step + 1) % gradient_accumulation_steps == 0:
                            # Clip gradients
                            if clip_grad_norm > 0:
                                if scaler is not None:
                                    scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model_components['unet'].parameters(), clip_grad_norm)
                            
                            # Optimizer step
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            
                            scheduler.step()
                            optimizer.zero_grad()
                        
                        # Update progress
                        step += 1
                        current_loss = loss.item() * gradient_accumulation_steps
                        epoch_loss += current_loss
                        
                        if current_loss < self.best_loss:
                            self.best_loss = current_loss
                        
                        self.current_loss = current_loss
                        self.progress = 20 + int((step / num_training_steps) * 75)
                        
                        # Log progress
                        if step % 10 == 0:
                            print(f"Step {step}/{num_training_steps}, Loss: {current_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")
                        
                    except Exception as e:
                        print(f"Error during training step {step}: {e}")
                        return (f"Training failed at step {step}: {str(e)}", f"{self.progress}%", f"{self.current_loss:.6f}", "")
                
                # Epoch completed
                avg_epoch_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{num_epochs} completed. Average loss: {avg_epoch_loss:.6f}")
                
                # Save checkpoint if needed
                if (epoch + 1) % kwargs.get('save_every_n_epochs', 5) == 0:
                    checkpoint_path = os.path.join(self.loras_path, f"{output_name}_epoch_{epoch + 1}.safetensors")
                    try:
                        self._save_lora_weights(model_components, checkpoint_path, {
                            'base_model': model_info['name'],
                            'trigger_word': trigger_word,
                            'rank': rank,
                            'alpha': alpha,
                            'final_loss': avg_epoch_loss,
                            'num_epochs': epoch + 1,
                            'learning_rate': learning_rate
                        })
                        print(f"Checkpoint saved: {checkpoint_path}")
                    except Exception as e:
                        print(f"Warning: Could not save checkpoint: {e}")
            
            # Training completed
            self.training_status = "saving"
            self.progress = 95
            
            # Save final LoRA
            output_dir = self._ensure_lora_output_directory()
            final_output_path = os.path.join(output_dir, f"{output_name}.safetensors")
            
            try:
                self._save_lora_weights(model_components, final_output_path, {
                    'base_model': model_info['name'],
                    'trigger_word': trigger_word,
                    'rank': rank,
                    'alpha': alpha,
                    'final_loss': self.current_loss,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate
                })
            except Exception as e:
                return (f"Error saving LoRA: {str(e)}", f"{self.progress}%", f"{self.current_loss:.6f}", "")
            
            # Cleanup
            self.training_status = "completed"
            self.progress = 100
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return (
                f"✅ LoRA training completed! Final loss: {self.current_loss:.6f}, Best loss: {self.best_loss:.6f}. Trained for {step} steps.",
                "100%",
                f"{self.current_loss:.6f}",
                final_output_path
            )
            
        except Exception as e:
            error_msg = f"LoRA training failed: {str(e)}"
            print(f"LoRA Training Error: {error_msg}")
            traceback.print_exc()
            
            # Clear GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return (
                error_msg,
                f"{self.progress}%",
                f"{self.current_loss:.6f}",
                ""
            )

    def _get_model_info(self, model_display_name: str) -> Dict[str, Any]:
        """Get model information from display name"""
        models = self._scan_available_models()
        
        for model in models:
            # Match display name format
            if model['model_type'] != 'Unknown':
                if model['category'] in ['checkpoint', 'unet']:
                    display_name = f"[{model['model_type']}] {model['name']} ({model['size_gb']:.1f}GB)"
                else:
                    display_name = f"[{model['model_type']}] {model['category']}/{model['name']} ({model['size_gb']:.1f}GB)"
            else:
                if model['category'] in ['checkpoint', 'unet']:
                    display_name = f"[{model['type'].upper()}] {model['name']} ({model['size_gb']:.1f}GB)"
                else:
                    display_name = f"[{model['type'].upper()}] {model['category']}/{model['name']} ({model['size_gb']:.1f}GB)"
            
            if display_name == model_display_name:
                return model
        
        return None

    def _prepare_dataset(self, dataset_path: str, trigger_word: str, resolution: int) -> Optional[LoRADataset]:
        """Prepare dataset for LoRA training"""
        try:
            # Find all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext}")))
                image_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext.upper()}")))
            
            if not image_files:
                print(f"No image files found in {dataset_path}")
                return None
            
            # Create dataset with images and captions
            dataset_items = []
            for img_path in image_files:
                # Look for corresponding caption file
                caption_path = None
                base_name = os.path.splitext(img_path)[0]
                
                for ext in ['.txt', '.caption']:
                    test_path = base_name + ext
                    if os.path.exists(test_path):
                        caption_path = test_path
                        break
                
                # Read caption or use trigger word
                if caption_path and os.path.exists(caption_path):
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                else:
                    caption = trigger_word
                
                # Add trigger word if not present
                if trigger_word not in caption:
                    caption = f"{trigger_word}, {caption}"
                
                dataset_items.append({
                    'image_path': img_path,
                    'caption': caption
                })
            
            print(f"Found {len(dataset_items)} images for training")
            
            # Create a simple tokenizer for now
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            
            return LoRADataset(dataset_items, tokenizer, resolution)
            
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return None

    def _load_base_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load the base model for training"""
        try:
            model_path = model_info['path']
            model_type = model_info['model_type']
            
            if model_type == 'SDXL':
                # Load SDXL pipeline
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                
                return {
                    'unet': pipe.unet,
                    'text_encoder': pipe.text_encoder,
                    'text_encoder_2': pipe.text_encoder_2,
                    'tokenizer': pipe.tokenizer,
                    'tokenizer_2': pipe.tokenizer_2,
                    'vae': pipe.vae,
                    'scheduler': pipe.scheduler,
                    'pipeline': pipe,
                    'model_type': 'SDXL'
                }
                
            elif model_type == 'SD1.5':
                # Load SD1.5 pipeline
                pipe = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                
                return {
                    'unet': pipe.unet,
                    'text_encoder': pipe.text_encoder,
                    'tokenizer': pipe.tokenizer,
                    'vae': pipe.vae,
                    'scheduler': pipe.scheduler,
                    'pipeline': pipe,
                    'model_type': 'SD1.5'
                }
            
            else:
                raise ValueError(f"Model type {model_type} not supported")
                
        except Exception as e:
            print(f"Error loading base model: {e}")
            raise

    def _inject_lora_layers(self, model_components: Dict[str, Any], rank: int, alpha: float, target_modules: List[str]) -> Dict[str, Any]:
        """Inject LoRA layers into the model"""
        try:
            # Create LoRA config
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.DIFFUSION_IMAGE_GENERATION,
            )
            
            # Apply LoRA to UNet
            model_components['unet'] = get_peft_model(model_components['unet'], lora_config)
            
            # Enable gradient checkpointing for memory efficiency
            model_components['unet'].enable_gradient_checkpointing()
            
            print(f"LoRA layers injected with rank={rank}, alpha={alpha}")
            return model_components
            
        except Exception as e:
            print(f"Error injecting LoRA layers: {e}")
            raise

    def _prepare_optimizer_and_scheduler(self, model_components: Dict[str, Any], learning_rate: float, optimizer_name: str, scheduler_name: str, num_training_steps: int, warmup_steps: int):
        """Prepare optimizer and scheduler for training"""
        try:
            # Get trainable parameters
            params_to_optimize = list(filter(lambda p: p.requires_grad, model_components['unet'].parameters()))
            
            # Create optimizer
            if optimizer_name == "AdamW":
                optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
            elif optimizer_name == "AdamW8bit":
                try:
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.AdamW8bit(params_to_optimize, lr=learning_rate)
                except ImportError:
                    print("Warning: bitsandbytes not available, using regular AdamW")
                    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
            elif optimizer_name == "Lion":
                try:
                    from lion_pytorch import Lion
                    optimizer = Lion(params_to_optimize, lr=learning_rate)
                except ImportError:
                    print("Warning: lion_pytorch not available, using AdamW")
                    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
            else:
                optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
            
            # Create scheduler
            scheduler = get_scheduler(
                scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
            
            return optimizer, scheduler
            
        except Exception as e:
            print(f"Error preparing optimizer/scheduler: {e}")
            raise

    def _compute_loss(self, model_components: Dict[str, Any], batch: Dict[str, torch.Tensor], noise_offset: float = 0.0) -> torch.Tensor:
        """Compute the training loss"""
        try:
            # Get model components
            unet = model_components['unet']
            vae = model_components['vae']
            text_encoder = model_components['text_encoder']
            scheduler = model_components['scheduler']
            
            # Get batch data
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            
            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            if noise_offset > 0:
                noise += noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
            
            # Sample timesteps
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to latents
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            # Encode text
            with torch.no_grad():
                if model_components['model_type'] == 'SDXL':
                    # SDXL has two text encoders
                    encoder_hidden_states = text_encoder(input_ids)[0]
                    # For simplicity, we'll use the first text encoder
                    # In a full implementation, you'd also use text_encoder_2
                else:
                    encoder_hidden_states = text_encoder(input_ids)[0]
            
            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            
            # Compute loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            return loss
            
        except Exception as e:
            print(f"Error computing loss: {e}")
            raise

    def _save_lora_weights(self, model_components: Dict[str, Any], output_path: str, metadata: Dict[str, Any]):
        """Save LoRA weights to file"""
        try:
            # Extract LoRA weights
            lora_state_dict = {}
            
            for name, param in model_components['unet'].named_parameters():
                if 'lora_' in name:
                    lora_state_dict[name] = param.detach().cpu()
            
            # Add metadata
            metadata_dict = {
                'created_with': 'ComfyUI Globetrotter LoRA Trainer',
                'model_type': model_components['model_type'],
                'rank': str(metadata['rank']),
                'alpha': str(metadata['alpha']),
                'learning_rate': str(metadata['learning_rate']),
                'final_loss': str(metadata['final_loss']),
                'num_epochs': str(metadata['num_epochs']),
                'trigger_word': metadata['trigger_word'],
                'base_model': metadata['base_model']
            }
            
            # Save to safetensors format
            safetensors.torch.save_file(lora_state_dict, output_path, metadata=metadata_dict)
            
            print(f"LoRA weights saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving LoRA weights: {e}")
            raise

# Node registration
NODE_CLASS_MAPPINGS = {
    "LoRATrainerNode": LoRATrainerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRATrainerNode": "🎯 LoRA Trainer"
}
