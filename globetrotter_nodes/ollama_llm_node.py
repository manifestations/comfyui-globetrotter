import requests
import json
import os
import random

class OllamaLLMNode:
    """
    A ComfyUI node that enhances prompts using a local Ollama LLM.
    It dynamically loads styles and prompt instructions from JSON files,
    and constructs a detailed prompt for image generation with advanced
    creative controls and contextual awareness.
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

    @staticmethod
    def load_creative_enhancers():
        """Load creative enhancement options for more sophisticated prompts."""
        return {
            "moods": [
                "dreamy", "ethereal", "mystical", "dramatic", "serene", "dynamic", 
                "melancholic", "euphoric", "nostalgic", "futuristic", "vintage", 
                "surreal", "intimate", "epic", "whimsical", "dark", "bright", "moody"
            ],
            "times": [
                "dawn", "sunrise", "morning", "midday", "afternoon", "golden hour",
                "dusk", "sunset", "twilight", "night", "midnight", "blue hour"
            ],
            "weather": [
                "sunny", "partly cloudy", "overcast", "misty", "foggy", "rainy",
                "stormy", "snowy", "windy", "humid", "dry", "crisp"
            ],
            "textures": [
                "smooth", "rough", "silky", "velvet", "leather", "metallic",
                "wooden", "fabric", "glass", "stone", "organic", "synthetic"
            ],
            "color_schemes": [
                "monochromatic", "complementary", "analogous", "triadic", "split-complementary",
                "warm tones", "cool tones", "earth tones", "pastel", "neon", "muted", "vibrant"
            ],
            "artistic_techniques": [
                "chiaroscuro", "sfumato", "impasto", "pointillism", "crosshatching",
                "watercolor bleeding", "oil painting", "digital painting", "mixed media",
                "collage", "photocollage", "double exposure"
            ],
            "perspectives": [
                "bird's eye view", "worm's eye view", "dutch angle", "over the shoulder",
                "through objects", "reflection", "silhouette", "backlit", "side profile"
            ],
            "emotions": [
                "contemplative", "joyful", "melancholic", "determined", "peaceful",
                "intense", "playful", "mysterious", "confident", "vulnerable", "fierce", "gentle"
            ]
        }

    @staticmethod
    def load_advanced_styles():
        """Load advanced artistic and photographic styles."""
        return {
            "art_movements": [
                "impressionism", "expressionism", "surrealism", "cubism", "art nouveau",
                "art deco", "bauhaus", "minimalism", "pop art", "abstract expressionism",
                "romanticism", "realism", "hyperrealism", "fauvism", "dadaism"
            ],
            "photo_styles": [
                "street photography", "portrait photography", "landscape photography",
                "macro photography", "fine art photography", "documentary photography",
                "fashion photography", "conceptual photography", "still life photography",
                "architectural photography", "candid photography", "editorial photography"
            ],
            "render_styles": [
                "photorealistic", "stylized", "cartoon", "anime", "pixel art",
                "low poly", "cel shading", "pencil sketch", "ink drawing", "watercolor",
                "oil painting", "digital art", "3D render", "clay render", "wireframe"
            ]
        }

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input fields for the ComfyUI node."""
        try:
            # Define base paths
            base_dir = os.path.dirname(__file__)
            data_dir = os.path.join(base_dir, '..', 'data')
            styles_dir = os.path.join(data_dir, 'styles')
            prompts_dir = os.path.join(data_dir, 'prompts')

            # Load creative enhancers
            creative_enhancers = cls.load_creative_enhancers()
            advanced_styles = cls.load_advanced_styles()

            # Load dynamic dropdowns from JSON files
            def load_style_list(filename):
                path = os.path.join(styles_dir, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        items = json.load(f)
                        return ['none', 'random'] + items if isinstance(items, list) else ['none', 'random']
                except Exception:
                    return ['none', 'random']

            cls.art_styles = cls.load_from_json(
                os.path.join(styles_dir, "art_styles.json")
            )
            cls.cameras = load_style_list("cameras.json")
            cls.film = load_style_list("film.json")
            cls.lighting = load_style_list("lighting.json")
            cls.photographers = load_style_list("photographers.json")
            cls.shot_types = load_style_list("shot_types.json")

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
                        ["none", "random"] + list(cls.art_styles.keys()),
                        {"default": "random"},
                    ),
                    "camera": (
                        cls.cameras,
                        {"default": "random"},
                    ),
                    "film": (
                        cls.film,
                        {"default": "random"},
                    ),
                    "lighting": (
                        cls.lighting,
                        {"default": "random"},
                    ),
                    "photographer": (
                        cls.photographers,
                        {"default": "random"},
                    ),
                    "shot_type": (
                        cls.shot_types,
                        {"default": "random"},
                    ),
                    # Creative Enhancement Controls
                    "mood": (
                        ["none", "random"] + creative_enhancers["moods"],
                        {"default": "none"},
                    ),
                    "time_of_day": (
                        ["none", "random"] + creative_enhancers["times"],
                        {"default": "none"},
                    ),
                    "weather": (
                        ["none", "random"] + creative_enhancers["weather"],
                        {"default": "none"},
                    ),
                    "texture": (
                        ["none", "random"] + creative_enhancers["textures"],
                        {"default": "none"},
                    ),
                    "color_scheme": (
                        ["none", "random"] + creative_enhancers["color_schemes"],
                        {"default": "none"},
                    ),
                    "art_movement": (
                        ["none", "random"] + advanced_styles["art_movements"],
                        {"default": "none"},
                    ),
                    "photo_style": (
                        ["none", "random"] + advanced_styles["photo_styles"],
                        {"default": "none"},
                    ),
                    "render_style": (
                        ["none", "random"] + advanced_styles["render_styles"],
                        {"default": "none"},
                    ),
                    "artistic_technique": (
                        ["none", "random"] + creative_enhancers["artistic_techniques"],
                        {"default": "none"},
                    ),
                    "perspective": (
                        ["none", "random"] + creative_enhancers["perspectives"],
                        {"default": "none"},
                    ),
                    "emotion": (
                        ["none", "random"] + creative_enhancers["emotions"],
                        {"default": "none"},
                    ),
                    "seed": (
                        "INT",
                        {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                    ),
                    "detail_scale": (
                        [
                            "none",
                            "exact",
                            "detailed",
                            "extra details",
                            "exaggerated",
                            "masterpiece",
                            "cinematic"
                        ],
                        {"default": "none"},
                    ),
                    "creative_mode": (
                        ["standard", "artistic", "photographic", "surreal", "minimalist"],
                        {"default": "standard"},
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
        camera="random",
        film="random",
        lighting="random",
        photographer="random",
        shot_type="random",
        mood="none",
        time_of_day="none",
        weather="none",
        texture="none",
        color_scheme="none",
        art_movement="none",
        photo_style="none",
        render_style="none",
        artistic_technique="none",
        perspective="none",
        emotion="none",
        detail_scale="none",
        creative_mode="standard",
        seed=0,
        ollama_url="http://127.0.0.1:11434/api/generate",
    ):
        if model_name == "disabled" or not keywords.strip():
            return (keywords,)

        # --- Add custom prompt if provided ---
        if custom_prompt.strip():
            keywords = f"{keywords.strip()}, {custom_prompt.strip()}"

        # --- Enhanced Randomization for Creative Controls ---
        creative_enhancers = self.load_creative_enhancers()
        advanced_styles = self.load_advanced_styles()
        
        def pick_random(option, options_list):
            if option == "random":
                filtered = [o for o in options_list if o not in ["none", "random"]]
                return random.choice(filtered) if filtered else "none"
            return option

        prompt_style = (
            random.choice(list(self.prompt_instructions.keys()))
            if prompt_style == "random" and self.prompt_instructions else prompt_style
        )
        art_style = pick_random(art_style, list(self.art_styles.keys()) if isinstance(self.art_styles, dict) else self.art_styles)
        camera = pick_random(camera, self.cameras)
        film = pick_random(film, self.film)
        lighting = pick_random(lighting, self.lighting)
        photographer = pick_random(photographer, self.photographers)
        shot_type = pick_random(shot_type, self.shot_types)
        
        # Randomize creative enhancement parameters
        mood = pick_random(mood, creative_enhancers["moods"])
        time_of_day = pick_random(time_of_day, creative_enhancers["times"])
        weather = pick_random(weather, creative_enhancers["weather"])
        texture = pick_random(texture, creative_enhancers["textures"])
        color_scheme = pick_random(color_scheme, creative_enhancers["color_schemes"])
        art_movement = pick_random(art_movement, advanced_styles["art_movements"])
        photo_style = pick_random(photo_style, advanced_styles["photo_styles"])
        render_style = pick_random(render_style, advanced_styles["render_styles"])
        artistic_technique = pick_random(artistic_technique, creative_enhancers["artistic_techniques"])
        perspective = pick_random(perspective, creative_enhancers["perspectives"])
        emotion = pick_random(emotion, creative_enhancers["emotions"])

        # --- Build Enhanced Prompt Based on Creative Mode ---
        final_keywords = keywords
        
        # Apply creative mode modifications
        if creative_mode == "artistic":
            if art_movement != "none":
                final_keywords += f", {art_movement} style"
            if artistic_technique != "none":
                final_keywords += f", {artistic_technique} technique"
        elif creative_mode == "photographic":
            if photo_style != "none":
                final_keywords += f", {photo_style}"
            if camera != "none":
                final_keywords += f", shot with {camera}"
        elif creative_mode == "surreal":
            final_keywords += ", surreal, dreamlike, fantastical elements"
            if mood != "none":
                final_keywords += f", {mood} atmosphere"
        elif creative_mode == "minimalist":
            final_keywords += ", clean, simple, minimal composition"
            if color_scheme != "none":
                final_keywords += f", {color_scheme}"

        # Add art style description if selected and available
        if art_style != "none" and art_style != "random":
            if isinstance(self.art_styles, dict) and art_style in self.art_styles:
                final_keywords += f", art style: {self.art_styles[art_style]}"
            elif isinstance(self.art_styles, list):
                final_keywords += f", art style: {art_style}"
                
        # Add render style
        if render_style != "none":
            final_keywords += f", rendered in {render_style}"
        # Add atmospheric and creative elements
        if mood != "none":
            final_keywords += f", {mood} mood"
        if emotion != "none":
            final_keywords += f", {emotion} expression"
        if time_of_day != "none":
            final_keywords += f", {time_of_day} lighting"
        if weather != "none":
            final_keywords += f", {weather} weather"
        if perspective != "none":
            final_keywords += f", {perspective}"
        if texture != "none":
            final_keywords += f", {texture} textures"
        if color_scheme != "none":
            final_keywords += f", {color_scheme} color palette"
            
        # Add traditional style options if not none
        if camera != "none" and creative_mode != "photographic":  # Avoid duplication
            final_keywords += f", camera: {camera}"
        if film != "none":
            final_keywords += f", film: {film}"
        if lighting != "none":
            final_keywords += f", lighting: {lighting}"
        if photographer != "none":
            final_keywords += f", photographer: {photographer}"
        if shot_type != "none":
            final_keywords += f", shot type: {shot_type}"

        # --- Enhanced Detail Scale System Prompt Modifier ---
        scale_instructions = {
            "none": "",
            "exact": "Strictly follow the user's prompt. Do not add extra details or embellishments.",
            "detailed": "Add more detail and description to the prompt, focusing on visual elements, textures, and atmosphere.",
            "extra details": "Add rich, vivid descriptions and creative flourishes to the prompt, expanding on implied visual elements.",
            "exaggerated": "Greatly exaggerate and elaborate on the prompt with highly imaginative, detailed descriptions and dramatic elements.",
            "masterpiece": "Transform this into a museum-quality masterpiece description with exceptional artistic detail, composition, and technical excellence.",
            "cinematic": "Create a cinematic, movie-quality scene with professional lighting, dramatic composition, and film-like atmosphere."
        }
        scale_text = scale_instructions.get(detail_scale, "")

        # --- Creative Mode System Prompts ---
        creative_mode_instructions = {
            "standard": "",
            "artistic": "Focus on artistic composition, color harmony, brushwork, and fine art techniques. Emphasize aesthetic beauty and artistic expression.",
            "photographic": "Emphasize photographic realism, lighting techniques, camera settings, and professional photography aesthetics.",
            "surreal": "Blend reality with fantastical elements. Create dreamlike, impossible, or metaphysical scenes that challenge perception.",
            "minimalist": "Focus on simplicity, clean lines, negative space, and essential elements. Remove unnecessary details for maximum impact."
        }
        
        creative_mode_text = creative_mode_instructions.get(creative_mode, "")

        # --- Always instruct the LLM to not output its own thoughts or meta-comments ---
        no_meta_instruction = (
            "Do not include any statements about what you think, your capabilities, or your role as an AI. "
            "Focus purely on creating a detailed, artistic prompt for image generation. "
            "Use vivid, descriptive language that would inspire a visual artist or AI image generator."
        )

        instructions = self.prompt_instructions.get(
            prompt_style, "Generate a detailed, creative prompt for image generation:"
        )
        
        # Compose the enhanced system prompt
        system_parts = [no_meta_instruction]
        if creative_mode_text:
            system_parts.append(creative_mode_text)
        if scale_text:
            system_parts.append(scale_text)
        system_parts.append(instructions)
        
        enhanced_instructions = "\n".join(system_parts)

        payload = {
            "model": model_name,
            "system": enhanced_instructions,
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
            # Filter out <think>...</think> blocks (DeepSeek workaround)
            import re
            final_prompt = re.sub(r'<think>.*?</think>', '', final_prompt, flags=re.DOTALL).strip()
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
