import requests
import json
import os
import base64
import io
from PIL import Image
import numpy as np
import torch

class OllamaVisionNode:
    """
    A ComfyUI node that describes images using a local Ollama LLM with vision capabilities.
    It takes an input image and generates detailed descriptions using advanced
    vision-language models with customizable description styles and detail levels.
    """
    
    @staticmethod
    def get_ollama_vision_models(ollama_url):
        """Fetches the list of vision-capable models from the Ollama API."""
        try:
            tags_url = ollama_url.replace("/api/generate", "/api/tags")
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            # Filter for vision-capable models (common vision models)
            vision_keywords = [
                "vision", "llava", "bakllava", "moondream", "cogvlm", 
                "blip", "instructblip", "minigpt", "mplug", "qwen-vl"
            ]
            
            vision_models = []
            for model in models:
                model_name = model["name"].lower()
                if any(keyword in model_name for keyword in vision_keywords):
                    vision_models.append(model["name"])
            
            # If no vision models found, return all models (user might have renamed them)
            if not vision_models:
                vision_models = [m["name"] for m in models]
                
            return vision_models
            
        except requests.exceptions.RequestException as e:
            print(f"OllamaVisionNode: Could not fetch Ollama models: {e}")
            return []

    @staticmethod
    def load_description_styles():
        """Load description style options for different use cases."""
        return {
            "analysis_depths": [
                "basic", "detailed", "comprehensive", "artistic", "technical", "creative"
            ],
            "focus_areas": [
                "general", "composition", "colors", "lighting", "mood", "subjects", 
                "background", "style", "technique", "emotions", "story", "details"
            ],
            "output_formats": [
                "natural", "structured", "bullet_points", "keywords", "narrative", "poetic"
            ],
            "perspectives": [
                "objective", "artistic", "photographic", "emotional", "technical", "storytelling"
            ]
        }

    @staticmethod
    def get_description_prompts():
        """Get system prompts for different description styles."""
        return {
            "basic": "Describe what you see in this image in simple, clear terms.",
            "detailed": "Provide a detailed description of this image, including all visible elements, composition, colors, and atmosphere.",
            "comprehensive": "Give a comprehensive analysis of this image covering composition, lighting, colors, subjects, mood, style, and any notable details.",
            "artistic": "Analyze this image from an artistic perspective, focusing on composition, technique, color harmony, and aesthetic elements.",
            "technical": "Provide a technical analysis of this image including lighting conditions, camera settings implications, composition techniques, and visual elements.",
            "creative": "Describe this image in a creative, imaginative way that captures its essence and tells a story about what you see.",
            "composition": "Focus on the compositional elements of this image: rule of thirds, leading lines, balance, symmetry, and visual flow.",
            "colors": "Analyze the color palette, color relationships, temperature, saturation, and how colors contribute to the mood.",
            "lighting": "Describe the lighting in this image: direction, quality, shadows, highlights, and atmospheric effects.",
            "mood": "Focus on the emotional impact and mood of this image, describing the feelings it evokes.",
            "subjects": "Describe the main subjects in this image, their poses, expressions, clothing, and interactions.",
            "background": "Focus on the background elements, setting, environment, and how they complement the main subject.",
            "style": "Analyze the artistic or photographic style of this image, including genre, technique, and influences.",
            "technique": "Focus on the technical aspects: depth of field, exposure, composition rules, and photographic techniques used.",
            "emotions": "Describe the emotional content of the image, including facial expressions, body language, and overall feeling.",
            "story": "Tell the story this image conveys, including narrative elements, implied action, and context.",
            "details": "Focus on fine details, textures, patterns, and small elements that might be easily overlooked."
        }

    @staticmethod
    def tensor_to_base64(tensor):
        """Convert a tensor image to base64 string."""
        try:
            # Handle different tensor formats
            if tensor.dim() == 4:  # Batch dimension
                tensor = tensor.squeeze(0)
            
            # Convert from [C, H, W] to [H, W, C] if needed
            if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0)
            
            # Ensure tensor is in [0, 1] range
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            
            # Convert to numpy
            np_image = tensor.cpu().numpy()
            
            # Handle grayscale
            if np_image.shape[-1] == 1:
                np_image = np.repeat(np_image, 3, axis=-1)
            
            # Convert to PIL Image
            pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            print(f"OllamaVisionNode: Error converting tensor to base64: {e}")
            return None

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input fields for the ComfyUI node."""
        try:
            # Get available vision models
            installed_models = cls.get_ollama_vision_models(
                "http://127.0.0.1:11434/api/generate"
            )
            
            # Load description styles
            prompts = cls.get_description_prompts()
            
            return {
                "required": {
                    "image": ("IMAGE",),
                    "model_name": (
                        ["disabled"] + installed_models,
                        {"default": "disabled"},
                    ),
                    "description_style": (
                        ["none", "random"] + list(prompts.keys()),
                        {"default": "detailed"},
                    ),
                    "custom_prompt": (
                        "STRING",
                        {"multiline": True, "default": ""},
                    ),
                },
                "optional": {
                    "ollama_url": (
                        "STRING",
                        {"default": "http://127.0.0.1:11434/api/generate"},
                    ),
                },
            }
        except Exception as e:
            print(f"OllamaVisionNode: CRITICAL ERROR in INPUT_TYPES: {e}")
            return {
                "required": {
                    "image": ("IMAGE",),
                    "error_message": (
                        "STRING",
                        {
                            "multiline": True,
                            "default": f"ERROR: Node failed to load. Check console for details: {e}",
                        },
                    ),
                }
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "describe_image"
    CATEGORY = "🌐 Globetrotter/Vision"
    NAME = "Ollama Vision Describer"

    def describe_image(
        self,
        image,
        model_name="disabled",
        description_style="detailed",
        custom_prompt="",
        ollama_url="http://127.0.0.1:11434/api/generate",
    ):
        """Generate a description of the input image using Ollama vision model."""
        
        if model_name == "disabled":
            return ("Vision model is disabled. Please select a vision-capable model.",)
        
        # If description_style is "none", return a simple default description
        if description_style == "none":
            return ("Please select a description style other than 'none' to generate an image description.",)
        
        # Convert tensor to base64
        img_base64 = self.tensor_to_base64(image)
        if img_base64 is None:
            return ("Error: Could not process the input image.",)
        
        # Build the system prompt
        prompts = self.get_description_prompts()
        
        # Handle random selection
        if description_style == "random":
            import random
            available_styles = [key for key in prompts.keys() if key != "none"]
            description_style = random.choice(available_styles) if available_styles else "detailed"
        
        # Get base prompt
        base_prompt = prompts.get(description_style, prompts["detailed"])
        
        # Add custom prompt if provided
        system_prompt = base_prompt
        if custom_prompt.strip():
            system_prompt += f"\n\nCustom instructions: {custom_prompt.strip()}"
        
        # Add general instructions
        system_prompt += "\n\nProvide your response in clear, descriptive language that would be useful for image generation or cataloging purposes."
        
        # Prepare the API payload
        payload = {
            "model": model_name,
            "prompt": "Describe this image in detail.",
            "system": system_prompt,
            "images": [img_base64],
            "stream": False,
            "options": {
                "num_predict": 500,
                "temperature": 0.7,
            }
        }
        
        # Make API call
        try:
            response = requests.post(ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            response_json = response.json()
            
            description = response_json.get("response", "").strip()
            
            # Clean up the response
            description = description.strip('"').strip("'").strip()
            
            if not description:
                return ("Error: Received empty response from the vision model.",)
            
            print(f"OllamaVisionNode: Generated description: {description[:100]}...")
            return (description,)
            
        except requests.exceptions.RequestException as e:
            error_message = f"OllamaVisionNode Error: Could not connect to Ollama. Details: {e}"
            print(error_message)
            return (f"ERROR: OLLAMA NOT REACHABLE. {error_message}",)
        except Exception as e:
            error_message = f"OllamaVisionNode Error: An unexpected error occurred: {e}"
            print(error_message)
            return (f"ERROR: UNEXPECTED. {error_message}",)


NODE_CLASS_MAPPINGS = {
    "OllamaVisionNode": OllamaVisionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVisionNode": "👁️ Ollama Vision Describer"
}
