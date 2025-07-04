import requests
import json
import os
import random

class OllamaLLMNode:
    """
    A ComfyUI node that enhances prompts using a local Ollama LLM.
    It dynamically loads styles and prompt instructions from JSON files,
    and constructs a detailed prompt for image generation.
    """
    
    @staticmethod
    def get_ollama_models(ollama_url):
        """Fetches the list of installed models from the Ollama API."""
        try:
            tags_url = ollama_url.replace("/api/generate", "/api/tags")
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except requests.exceptions.RequestException as e:
            print(f"OllamaLLMNode: Could not fetch Ollama models: {e}")
            return []

    @staticmethod
    def load_from_json(file_path):
        """A robust utility to load a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"OllamaLLMNode: Error loading {file_path}: {e}")
            return {}

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input fields for the ComfyUI node."""
        try:
            # Define base paths
            base_dir = os.path.dirname(__file__)
            data_dir = os.path.join(base_dir, '..', 'data')
            styles_dir = os.path.join(data_dir, 'styles')
            prompts_dir = os.path.join(data_dir, 'prompts')

            # Load dynamic dropdowns from JSON files
            cls.art_styles = cls.load_from_json(
                os.path.join(styles_dir, "art_styles.json")
            )

            # Load prompt instructions
            cls.prompt_instructions = {}
            if os.path.exists(prompts_dir):
                for f in os.listdir(prompts_dir):
                    if f.endswith('.json'):
                        style_name = os.path.splitext(f)[0]
                        content = cls.load_from_json(
                            os.path.join(prompts_dir, f)
                        )
                        if "instructions" in content:
                            cls.prompt_instructions[style_name] = content[
                                "instructions"
                            ]

            installed_models = cls.get_ollama_models(
                "http://127.0.0.1:11434/api/generate"
            )

            return {
                "required": {
                    "keywords": (
                        "STRING",
                        {
                            "multiline": True,
                            "default": "a cat, sitting on a chair, fantasy art",
                        },
                    ),
                    "custom_prompt": (
                        "STRING",
                        {"multiline": True, "default": ""},
                    ),
                    "model_name": (
                        ["disabled"] + installed_models,
                        {"default": "disabled"},
                    ),
                    "prompt_style": (
                        ["random", "disabled"]
                        + list(cls.prompt_instructions.keys()),
                        {"default": "random"},
                    ),
                    "art_style": (
                        ["random", "disabled"] + list(cls.art_styles.keys()),
                        {"default": "random"},
                    ),
                    "seed": (
                        "INT",
                        {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
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
            print(f"OllamaLLMNode: CRITICAL ERROR in INPUT_TYPES: {e}")
            # Return a degraded input that shows the error in the UI
            return {
                "required": {
                    "keywords": (
                        "STRING",
                        {
                            "multiline": True,
                            "default": f"ERROR: Node failed to load. Check console for details: {e}",
                        },
                    ),
                }
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "🌐 Globetrotter/LLM"
    NAME = "Ollama Prompter"

    def generate_prompt(
        self,
        keywords,
        custom_prompt="",
        model_name="disabled",
        prompt_style="random",
        art_style="random",
        seed=0,
        ollama_url="http://127.0.0.1:11434/api/generate",
    ):
        if model_name == "disabled" or not keywords.strip():
            return (keywords,)

        # --- Add custom prompt if provided ---
        if custom_prompt.strip():
            keywords = f"{keywords.strip()}, {custom_prompt.strip()}"

        # --- Randomization ---
        if prompt_style == "random":
            prompt_style = (
                random.choice(list(self.prompt_instructions.keys()))
                if self.prompt_instructions
                else "disabled"
            )
        if art_style == "random":
            art_style = (
                random.choice(list(self.art_styles.keys()))
                if self.art_styles
                else "disabled"
            )

        # --- Build the prompt for the LLM ---
        final_keywords = keywords
        if art_style != "disabled" and art_style in self.art_styles:
            final_keywords += f", art style: {self.art_styles[art_style]}"

        instructions = self.prompt_instructions.get(
            prompt_style, "Generate a detailed prompt:"
        )

        payload = {
            "model": model_name,
            "system": instructions,
            "prompt": final_keywords,
            "stream": False,
            "options": {"seed": seed}
        }

        # --- API Call ---
        try:
            response = requests.post(ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            
            final_prompt = (
                response_json.get("response", "").strip().strip('"').strip("'")
            )
            print(f"OllamaLLMNode: Received prompt: {final_prompt}")
            return (final_prompt,)

        except requests.exceptions.RequestException as e:
            error_message = (
                f"OllamaLLMNode Error: Could not connect to Ollama. Details: {e}"
            )
            print(error_message)
            return (f"ERROR: OLLAMA NOT REACHABLE. Keywords: {keywords}",)
        except Exception as e:
            error_message = (
                f"OllamaLLMNode Error: An unexpected error occurred: {e}"
            )
            print(error_message)
            return (f"ERROR: UNEXPECTED. Keywords: {keywords}",)


NODE_CLASS_MAPPINGS = {
    "OllamaLLMNode": OllamaLLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaLLMNode": "✨ Ollama Prompter"
}
