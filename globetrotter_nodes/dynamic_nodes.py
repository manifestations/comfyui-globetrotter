import os
import json
import random

# --- Constants ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
COUNTRIES_FILE = os.path.join(DATA_DIR, 'countries.json')
ATTIRE_DIR = os.path.join(DATA_DIR, 'attire')
GENERIC_HAIR_FILE = os.path.join(ATTIRE_DIR, 'head', 'hair', 'generic.json')

# JSON Data Files
WEATHER_MOODS_FILE = os.path.join(DATA_DIR, 'ui', 'weather_moods.json')
SMART_DEFAULTS_FILE = os.path.join(DATA_DIR, 'generation', 'smart_defaults.json')
CONTEXTUAL_SUGGESTIONS_FILE = os.path.join(DATA_DIR, 'generation', 'contextual_suggestions.json')
AI_PROMPT_STRUCTURE_FILE = os.path.join(DATA_DIR, 'generation', 'ai_prompt_structure.json')
DYNAMIC_NODE_OPTIONS_FILE = os.path.join(DATA_DIR, 'ui', 'dynamic_node_options.json')
COMPLEMENTARY_COLORS_FILE = os.path.join(DATA_DIR, 'ui', 'complementary_colors.json')
GENDER_CONFIG_FILE = os.path.join(DATA_DIR, 'config', 'gender_config.json')
PROMPT_CONFIG_FILE = os.path.join(DATA_DIR, 'config', 'prompt_config.json')

# --- Helper Functions ---

def load_json_file(file_path: str, default_value=None):
    """Load a JSON file with error handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return default_value or {}

def find_attire_parts(directory):
    """
    Recursively finds all body part directories that contain attire JSON files.
    Returns a sorted list of unique relative paths.
    """
    attire_paths = set()
    for root, _, files in os.walk(directory):
        if any(f.endswith('.json') for f in files):
            relative_path = os.path.relpath(root, directory).replace('\\', '/')
            if relative_path != '.':
                attire_paths.add(relative_path)
    return sorted(list(attire_paths))

def load_attire_options(country_code, gender_filter=None):
    """
    Loads attire options for a specific country with optional gender filtering.
    """
    country_attire_options = {}
    gender_config = load_json_file(GENDER_CONFIG_FILE, {})
    
    # Determine valid genders for filtering
    valid_genders = []
    if gender_filter:
        gender_mappings = gender_config.get("gender_mappings", {})
        valid_genders = gender_mappings.get(gender_filter.lower(), [gender_filter.lower()])

    # Iterate through all JSON files in the attire directory for the given country
    for root, _, files in os.walk(ATTIRE_DIR):
        for file in files:
            if file == f"{country_code}.json":
                json_path = os.path.join(root, file)
                data = load_json_file(json_path)
                if data:
                    body_part = data.get("body_part")
                    if body_part and "attires" in data:
                        filtered_attires = []
                        
                        for attire in data["attires"]:
                            # Check if attire matches gender filter
                            if gender_filter and valid_genders:
                                attire_genders = attire.get("gender", [])
                                if isinstance(attire_genders, str):
                                    attire_genders = [attire_genders]
                                
                                # Check if any of the attire genders match valid genders
                                gender_match = any(
                                    gender.lower() in valid_genders 
                                    for gender in attire_genders
                                )
                                
                                if not gender_match:
                                    continue
                            
                            filtered_attires.append(attire["name"])
                        
                        if filtered_attires:
                            options = ["none", "random"] + filtered_attires
                            # Capitalize each word in each option except 'none' and 'random'
                            options = [
                                opt if opt in ["none", "random"] else opt.title()
                                for opt in options
                            ]
                            country_attire_options[body_part.title()] = options
    
    return country_attire_options

def load_appearance_options(country_code: str):
    """Load region appearance options for a country from
    data/appearance/<country>.json."""
    appearance_file = os.path.join(
        DATA_DIR, "appearance", f"{country_code}.json"
    )
    if not os.path.exists(appearance_file):
        return []
    
    data = load_json_file(appearance_file, [])
    if not data:
        return []
    
    # Capitalize each word in each region name
    regions = data.get("regions", [])
    for region in regions:
        if "name" in region:
            region["name"] = region["name"].title()
    return regions

def load_generic_hair_styles():
    """Load generic hair styles from generic.json."""
    data = load_json_file(GENERIC_HAIR_FILE, [])
    if not data:
        return ["none", "random"]
    
    options = ["none", "random"] + data
    return options

def load_pose_options(country_code: str):
    """Load pose options for a country from data/poses/<country>.json."""
    pose_file = os.path.join(
        DATA_DIR, "poses", f"{country_code}.json"
    )
    if not os.path.exists(pose_file):
        return ["none", "random"]
    
    data = load_json_file(pose_file, {})
    if not data:
        return ["none", "random"]
    
    poses = data.get("poses", [])
    return ["none", "random"] + [p.title() for p in poses]

def load_cultural_context(country_code: str):
    """Load cultural activities, festivals, and context for a country."""
    cultural_file = os.path.join(
        DATA_DIR, "cultural", f"{country_code}.json"
    )
    if not os.path.exists(cultural_file):
        return {"activities": [], "festivals": [], "landmarks": [], "colors": []}
    
    return load_json_file(cultural_file, {"activities": [], "festivals": [], "landmarks": [], "colors": []})

def load_weather_moods():
    """Load weather and lighting conditions for more atmospheric prompts."""
    return load_json_file(WEATHER_MOODS_FILE, {
        "weather": ["sunny", "cloudy", "rainy", "misty"],
        "lighting": ["soft natural light", "dramatic shadows"],
        "emotions": ["serene", "joyful", "contemplative", "confident"],
        "settings": ["urban street", "traditional market", "temple grounds"]
    })

def get_complementary_colors(region_info):
    """Get colors that complement the region's traditional palette."""
    if not region_info:
        return []
    
    color_data = load_json_file(COMPLEMENTARY_COLORS_FILE, {})
    if not color_data:
        return []
    
    # Map common skin tones to complementary colors
    skin_tone = region_info.get('skin_tone', '').lower()
    color_map = color_data.get('color_map', {})
    
    for tone in color_map:
        if tone in skin_tone:
            return color_map[tone]
    return color_data.get('default_colors', ['earth tones', 'natural colors'])

def format_attire_list(attires: list[str]) -> str:
    """Format a list of attires in natural language (comma and 'and')."""
    if not attires:
        return ""
    if len(attires) == 1:
        return attires[0]
    return ", ".join(attires[:-1]) + " and " + attires[-1]

def load_smart_defaults():
    """Load intelligent defaults that work well together for prompt generation."""
    return load_json_file(SMART_DEFAULTS_FILE, {
        "age_gender_combos": {},
        "optimal_combinations": {}
    })

def load_contextual_suggestions():
    """Load contextual suggestions that improve prompt coherence."""
    return load_json_file(CONTEXTUAL_SUGGESTIONS_FILE, {
        "weather_activity_pairs": {},
        "emotion_setting_pairs": {},
        "cultural_context_suggestions": {},
        "attire_activity_pairs": {}
    })

def get_ai_optimized_prompt_structure():
    """Return prompt structure optimized for AI image generation."""
    return load_json_file(AI_PROMPT_STRUCTURE_FILE, {
        "priority_order": ["subject_description"],
        "ai_friendly_keywords": {"quality": [], "lighting": [], "composition": [], "style": []},
        "negative_prompt_hints": {"general": [], "portraits": [], "landscapes": []}
    })

def load_dynamic_node_options():
    """Load UI options for dynamic nodes."""
    return load_json_file(DYNAMIC_NODE_OPTIONS_FILE, {
        "age": ["young adult", "adult"],
        "gender": ["Female", "Male"],
        "emotion": {"prioritized": ["confident"], "default": "confident"},
        "detail_level": {"options": ["detailed"], "default": "detailed"},
        "composition": {"options": ["rule of thirds"], "default": "rule of thirds"}
    })

def load_gender_config():
    """Load gender configuration for filtering."""
    return load_json_file(GENDER_CONFIG_FILE, {
        "gender_mappings": {
            "male": ["male", "unisex"],
            "female": ["female", "unisex"],
            "non-binary": ["unisex"],
            "unspecified": ["male", "female", "unisex"]
        }
    })

def load_prompt_config():
    """Load prompt configuration for technical enhancements."""
    return load_json_file(PROMPT_CONFIG_FILE, {
        "technical_enhancements": {},
        "detail_prefixes": {},
        "llm_rewrite_settings": {}
    })

# --- Pre-load static data ---
COUNTRIES = load_json_file(COUNTRIES_FILE, [])
if not COUNTRIES:
    print("Fatal Error: Could not load countries.json")

ALL_ATTIRE_PARTS = find_attire_parts(ATTIRE_DIR)

# --- Dynamic Node Generation ---
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from transformers import pipeline
    llm_rewriter = pipeline("text-generation", model="distilgpt2", device=-1)
    LLM_AVAILABLE = True
except ImportError:
    llm_rewriter = None
    LLM_AVAILABLE = False


def create_attire_node(country: dict) -> tuple[str, type]:
    """Factory function to create a dynamic node class for a country."""
    country_code = country['code']
    country_name = country['name']
    class_name = f"{country_name.replace(' ', '')}AttireNode"

    # Pre-load base data for this specific country
    region_options = load_appearance_options(country_code)
    region_names = [r['name'] for r in region_options] if region_options else []
    generic_hair_styles = load_generic_hair_styles()
    pose_options = load_pose_options(country_code)
    cultural_context = load_cultural_context(country_code)
    weather_moods = load_weather_moods()
    smart_defaults = load_smart_defaults()
    contextual_suggestions = load_contextual_suggestions()
    ai_prompt_structure = get_ai_optimized_prompt_structure()
    dynamic_options = load_dynamic_node_options()
    gender_config = load_gender_config()
    prompt_config = load_prompt_config()

    def create_input_types():
        """Create input types with access to country_code from closure."""
        def ensure_none_random(options_list, default_value=None):
            """Ensure every dropdown has 'none' and 'random' options at the start, and Title Case for display.
            Also ensures the default value is present in the list."""
            if not options_list:
                base = ["none", "random"]
                if default_value and default_value not in base:
                    base.append(default_value)
                return base
            # Convert to list if it's a set or other iterable
            if not isinstance(options_list, list):
                options_list = list(options_list)
            # Remove existing 'none' and 'random' entries (case-insensitive)
            filtered_options = [opt for opt in options_list if opt.lower() not in ['none', 'random']]
            # Title Case for display, except 'none' and 'random'
            filtered_options = [opt if opt.lower() in ['none', 'random'] else opt.title() for opt in filtered_options]
            # Ensure default value is present
            if default_value and default_value not in filtered_options and default_value not in ["none", "random"]:
                filtered_options.append(default_value)
            return ["none", "random"] + filtered_options
        
        # Smart ordering for better UX and AI generation
        required = {
            # Primary Subject Definition (most important for AI)
            "age": (ensure_none_random(dynamic_options.get("age", ["young adult", "adult"]), default_value="young adult"), {"default": "young adult"}),
            "gender": (ensure_none_random(dynamic_options.get("gender", ["Female", "Male"]), default_value="Female"), {"default": "Female"}),
            "region": (ensure_none_random(region_names if region_names else ["Unspecified"]), {"default": "none"}),
            
            # Physical Characteristics  
            "hair_style": (generic_hair_styles, {"default": "none"}),
            "emotion": (
                ensure_none_random(
                    dynamic_options.get("emotion", {}).get("prioritized", ["confident"]) + 
                    [e for e in weather_moods.get("emotions", []) if e not in dynamic_options.get("emotion", {}).get("prioritized", [])]
                ),
                {"default": dynamic_options.get("emotion", {}).get("default", "confident")}
            ),
            
            # Actions and Context
            "pose": (pose_options, {"default": "none"}),
            "cultural_element": (
                ensure_none_random(
                    sorted(cultural_context.get("activities", []) + cultural_context.get("festivals", []), 
                           key=lambda x: len(x))  # Sort by length for better UX
                ),
                {"default": "none"}
            ),
            
            # Environment and Atmosphere
            "setting": (
                ensure_none_random(
                    # Prioritize settings that work well for AI
                    dynamic_options.get("setting", {}).get("prioritized", []) +
                    [s for s in weather_moods.get("settings", []) + cultural_context.get("landmarks", []) 
                     if s not in dynamic_options.get("setting", {}).get("prioritized", [])]
                ),
                {"default": "none"}
            ),
            "atmosphere": (
                ensure_none_random(
                    # Prioritize atmospheres that consistently produce good results
                    dynamic_options.get("atmosphere", {}).get("prioritized", []) +
                    [a for a in weather_moods.get("weather", []) + weather_moods.get("lighting", []) 
                     if a not in dynamic_options.get("atmosphere", {}).get("prioritized", [])]
                ),
                {"default": "golden hour"}
            ),
            
            # Technical and Style Controls
            "detail_level": (
                ensure_none_random(dynamic_options.get("detail_level", {}).get("options", ["detailed"])),
                {"default": dynamic_options.get("detail_level", {}).get("default", "detailed")}
            ),
            "composition": (
                ensure_none_random(dynamic_options.get("composition", {}).get("options", ["rule of thirds"])),
                {"default": dynamic_options.get("composition", {}).get("default", "rule of thirds")}
            ),
            "prompt_optimization": (
                ensure_none_random(dynamic_options.get("prompt_optimization", {}).get("options", ["ai_friendly"])),
                {"default": dynamic_options.get("prompt_optimization", {}).get("default", "ai_friendly")}
            ),
        }
        
        # Add attire options for all genders (they will be filtered in generate_prompt)
        # Load attire options for each gender to ensure all are available in UI
        all_attire_parts = {}
        for gender_option in dynamic_options.get("gender", ["Female", "Male"]):
            # Use the country_code from the factory function scope
            gender_attire = load_attire_options(country_code, gender_option)
            for body_part, options in gender_attire.items():
                if body_part not in all_attire_parts:
                    all_attire_parts[body_part] = set()
                all_attire_parts[body_part].update(options)
        
        # Add all unique attire options to the UI with consistent "none" and "random"
        for body_part, options_set in all_attire_parts.items():
            if options_set:
                # Convert set to list and ensure proper ordering
                options_list = list(options_set)
                options_list = ensure_none_random(options_list)
                required[body_part] = (options_list, {"default": "none"})
        
        # Simplified Controls (clearer and more intuitive)
        required.update({
            "randomization": (
                ["off", "light", "moderate", "full"],
                {
                    "default": "moderate",
                    "tooltip": "Controls how much randomization is applied: off=no randomization, light=minimal variation, moderate=balanced randomization, full=maximum creativity"
                }
            ),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "tooltip": "Random seed for reproducible results. Use same seed to get consistent outputs"}),
            "lora_trigger": (
                "STRING",
                {"multiline": False, "default": "", "tooltip": "Character name or trigger word for LoRA models"}
            ),
            "custom_prompt": (
                "STRING",
                {"multiline": True, "default": "", "tooltip": "Additional custom text to include in the prompt"}
            ),
            "experimental_llm_rewrite": (
                ['off', 'on'], 
                {"default": "off", "tooltip": "Experimental: Use AI to rewrite and enhance the prompt (requires transformers library)"}
            )
        })
        
        return {"required": required}

    @classmethod
    def input_types(cls):
        """Return optimized input types for efficient AI prompt generation."""
        return create_input_types()

    def generate_prompt(
        self,
        region: str,
        hair_style: str,
        pose: str,
        gender: str,
        age: str,
        randomization: str,
        seed: int,
        lora_trigger: str,
        custom_prompt: str,
        experimental_llm_rewrite: str = 'off',
        atmosphere: str = 'golden hour',
        emotion: str = 'confident',
        setting: str = 'none',
        cultural_element: str = 'none',
        detail_level: str = 'detailed',
        composition: str = 'rule of thirds',
        prompt_optimization: str = 'ai_friendly',
        **kwargs,
    ) -> tuple[str]:
        """Generate an AI-optimized natural language prompt."""
        
        # Load gender-filtered attire options
        country_attire_options = load_attire_options(country_code, gender)
        
        # Smart defaults for better AI generation
        if gender == "Unspecified":
            # Use age-appropriate gender defaults
            age_defaults = smart_defaults.get("age_gender_combos", {}).get(age, {})
            common_genders = age_defaults.get("common_genders", ["Female"])
            gender = common_genders[0] if common_genders else "Female"
        
        if age == "unspecified":
            age = "young adult"  # Most versatile for AI generation

        # Set up enhanced randomization with clearer options
        rng = random.Random()
        if randomization == "off":
            # No randomization - use fixed seed for consistency
            rng.seed(seed)
        elif randomization == "light":
            # Light randomization - some variation but mostly consistent
            rng.seed(seed)
        elif randomization == "moderate":
            # Moderate randomization - balanced variation with smart contextual choices
            rng.seed(seed)
        elif randomization == "full":
            # Full randomization - maximum creativity, less predictable
            # Use system random for more variation
            pass
        # else: system random (no seed set)

        # Intelligent parameter selection with contextual awareness
        def smart_random_selection(param, options_list, context=None):
            if param == "random":
                if context and randomization in ["moderate", "full"]:
                    # Use contextual suggestions for better combinations
                    if context == "atmosphere_for_setting" and setting != "none":
                        suitable_atmospheres = []
                        for weather, activities in contextual_suggestions.get("weather_activity_pairs", {}).items():
                            if any(activity in setting.lower() for activity in activities):
                                suitable_atmospheres.append(weather)
                        if suitable_atmospheres:
                            valid_options = [opt for opt in suitable_atmospheres if opt in options_list]
                            if valid_options:
                                return rng.choice(valid_options)
                    elif context == "emotion_for_setting" and setting != "none":
                        suitable_emotions = []
                        for emo, settings in contextual_suggestions.get("emotion_setting_pairs", {}).items():
                            if any(s in setting.lower() for s in settings):
                                suitable_emotions.append(emo)
                        if suitable_emotions:
                            valid_options = [opt for opt in suitable_emotions if opt in options_list]
                            if valid_options:
                                return rng.choice(valid_options)
                
                # Standard random selection (filtered by randomization level)
                filtered = [o for o in options_list if o not in ["none", "random"]]
                if filtered:
                    if randomization == "light":
                        # For light randomization, prefer first few options (more conservative)
                        conservative_options = filtered[:min(3, len(filtered))]
                        return rng.choice(conservative_options)
                    else:
                        # For moderate/full randomization, use all options
                        return rng.choice(filtered)
                else:
                    return "none"
            return param if param != "random" else "none"

        # Apply smart randomization
        atmosphere = smart_random_selection(atmosphere, weather_moods.get("weather", []) + weather_moods.get("lighting", []), "atmosphere_for_setting")
        emotion = smart_random_selection(emotion, weather_moods.get("emotions", []), "emotion_for_setting")
        setting = smart_random_selection(setting, weather_moods.get("settings", []) + cultural_context.get("landmarks", []))
        cultural_element = smart_random_selection(cultural_element, cultural_context.get("activities", []) + cultural_context.get("festivals", []))
        composition = smart_random_selection(composition, ["rule of thirds", "centered", "close-up", "wide shot", "portrait", "full body", "environmental"])

        # Handle region with intelligence
        region_val = region
        if region.lower() == "random" and region_names:
            region_val = rng.choice([r for r in region_names if r.lower() not in ["none", "random", "unspecified"]])
        elif region.lower() == "none":
            region_val = None

        # Validate region selection
        region_info = next(
            (r for r in region_options if r["name"] == region_val), None
        )
        region_display = (
            region_val
            if region_val and region_val != "Unspecified" and region_info
            else None
        )

        # AI-optimized prompt construction based on optimization mode
        prompt_parts = []
        
        # Get detail prefixes from configuration
        detail_prefixes = prompt_config.get("detail_prefixes", {}).get(prompt_optimization, {})
        if not detail_prefixes:
            # Fallback to basic prefixes
            detail_prefixes = {
                "basic": "",
                "detailed": "highly detailed, best quality, ",
                "cinematic": "cinematic masterpiece, professional photography, ",
                "artistic": "fine art masterpiece, museum quality, ",
                "photorealistic": "photorealistic, 8k ultra detailed, sharp focus, "
            }
        
        if prompt_optimization == "ai_friendly":
            # Structure optimized for AI understanding
            detail_prefix = detail_prefixes.get(detail_level, "")
            
            # Add quality enhancers first for AI models
            if detail_level != "basic":
                prompt_parts.extend([detail_prefix.rstrip(", ")])
            
            # 2. Main subject with composition
            if composition != "none":
                prompt_parts.append(f"{composition} composition")
            
            # 3. Subject identification
            if lora_trigger.strip():
                if region_display:
                    subject_desc = f"A {age} {gender} called {lora_trigger.strip()} from {region_display}, {country_name}"
                else:
                    subject_desc = f"A {age} {gender} called {lora_trigger.strip()} from {country_name}"
            else:
                if region_display:
                    subject_desc = f"A {age} {gender} from {region_display}, {country_name}"
                else:
                    subject_desc = f"A {age} {gender} from {country_name}"
            prompt_parts.append(subject_desc)
            
        elif prompt_optimization == "creative":
            # More artistic and creative structure
            if composition != "none":
                prompt_parts.append(f"{composition} composition")
                
            detail_prefix = detail_prefixes.get(detail_level, "")
            
            if lora_trigger.strip():
                if region_display:
                    prompt_parts.append(f"{detail_prefix}A {age} {gender} named {lora_trigger.strip()} from the {region_display} region of {country_name}")
                else:
                    prompt_parts.append(f"{detail_prefix}A {age} {gender} named {lora_trigger.strip()} from {country_name}")
            else:
                if region_display:
                    prompt_parts.append(f"{detail_prefix}A {age} {gender} from {region_display}, {country_name}")
                else:
                    prompt_parts.append(f"{detail_prefix}A {age} {gender} from {country_name}")
        
        else:  # balanced, technical, artistic
            # Standard structure with enhancements
            if composition != "none":
                prompt_parts.append(f"{composition} composition")
            
            detail_prefix = detail_prefixes.get(detail_level, "")
            
            if lora_trigger.strip():
                if region_display:
                    prompt_parts.append(f"{detail_prefix}A {age} {gender} called {lora_trigger.strip()} from {region_display}, {country_name}")
                else:
                    prompt_parts.append(f"{detail_prefix}A {age} {gender} called {lora_trigger.strip()} from {country_name}")
            else:
                if region_display:
                    prompt_parts.append(f"{detail_prefix}A {age} {gender} from {region_display}, {country_name}")
                else:
                    prompt_parts.append(f"{detail_prefix}A {age} {gender} from {country_name}")

        # Add emotional state (important for AI understanding)
        if emotion != "none":
            prompt_parts.append(f"with a {emotion} expression")

        # Add region-specific details
        if region_info:
            desc = region_info.get("description", "")
            if desc:
                prompt_parts.append(desc)
            prompt_parts.append(f"skin tone: {region_info.get('skin_tone','unspecified')}")
            
            # Add complementary colors for artistic enhancement
            if detail_level in ["artistic", "cinematic"]:
                comp_colors = get_complementary_colors(region_info)
                if comp_colors:
                    prompt_parts.append(f"color palette: {', '.join(comp_colors)}")
        
        # Hair styling
        if hair_style not in ["none", "random", "Unspecified", ""]:
            prompt_parts.append(f"hair: {hair_style}")
        elif region_info:
            prompt_parts.append(f"hair: {region_info.get('hair','unspecified')}")

        # Gender-filtered attire description with validation
        selected_attires = []
        for body_part, options in country_attire_options.items():
            selected = kwargs.get(body_part)
            if selected == "random":
                choosable = [opt for opt in options if opt not in ["none", "random"]]
                if choosable:
                    selected = rng.choice(choosable)
                else:
                    selected = "none"
            
            # Validate that selected attire is appropriate for the chosen gender
            if selected and selected != "none":
                attire_desc = None
                is_valid_for_gender = False
                
                # Find the correct attire file path for this body part
                attire_file = None
                for root, _, files in os.walk(ATTIRE_DIR):
                    for file in files:
                        if file == f"{country_code}.json":
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                    if data.get("body_part", "").lower() == body_part.lower():
                                        attire_file = file_path
                                        break
                            except Exception:
                                continue
                    if attire_file:
                        break
                
                if attire_file and os.path.exists(attire_file):
                    try:
                        with open(attire_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for attire in data.get("attires", []):
                                if attire["name"].lower() == selected.lower():
                                    attire_desc = attire.get("description")
                                    
                                    # Check if this attire is valid for the current gender
                                    attire_genders = attire.get("gender", [])
                                    if isinstance(attire_genders, str):
                                        attire_genders = [attire_genders]
                                    
                                    # Get valid genders for current selection
                                    gender_mappings = gender_config.get("gender_mappings", {})
                                    valid_genders = gender_mappings.get(gender.lower(), [gender.lower()])
                                    
                                    # Check if attire gender matches valid genders for current character
                                    is_valid_for_gender = any(
                                        ag.lower() in valid_genders 
                                        for ag in attire_genders
                                    )
                                    break
                    except Exception:
                        pass
                
                # Only add attire if it's valid for the current gender
                if is_valid_for_gender:
                    if attire_desc:
                        selected_attires.append(f"{selected}: {attire_desc}")
                    else:
                        selected_attires.append(selected)
                else:
                    # Log gender mismatch for debugging (optional)
                    print(f"Warning: Attire '{selected}' is not appropriate for gender '{gender}' - skipping")
        
        if selected_attires:
            prompt_parts.append("wearing " + format_attire_list(selected_attires))

        # Actions and context
        if pose not in ["none", "random", "Unspecified", ""]:
            prompt_parts.append(f"pose: {pose}")

        if cultural_element != "none":
            prompt_parts.append(f"cultural context: {cultural_element}")

        # Environment with AI-friendly keywords
        if setting != "none":
            prompt_parts.append(f"setting: {setting}")
        
        if atmosphere != "none":
            prompt_parts.append(f"atmosphere: {atmosphere}")

        # Custom additions
        if custom_prompt.strip():
            prompt_parts.append(custom_prompt.strip())

        # AI-optimized technical enhancements from configuration
        technical_enhancements = prompt_config.get("technical_enhancements", {})
        if detail_level in technical_enhancements:
            prompt_parts.append(technical_enhancements[detail_level])

        # Final assembly with AI optimization
        prompt = ", ".join([part for part in prompt_parts if part])
        prompt = prompt[:1].upper() + prompt[1:] if prompt else prompt
        
        # Apply prompt optimization post-processing
        if prompt_optimization == "ai_friendly":
            # Ensure key AI-friendly terms are present
            if detail_level != "basic" and "best quality" not in prompt.lower():
                prompt = "best quality, " + prompt
                
        # Experimental LLM rewrite with optimization awareness
        if experimental_llm_rewrite == 'on' and LLM_AVAILABLE and prompt:
            try:
                # Get max_length from configuration
                max_length_map = prompt_config.get("llm_rewrite_settings", {}).get("max_length_map", {
                    "ai_friendly": 60,
                    "creative": 80, 
                    "technical": 70,
                    "artistic": 75,
                    "balanced": 65
                })
                max_len = max_length_map.get(prompt_optimization, 65)
                
                result = llm_rewriter(prompt, max_length=min(len(prompt.split()) + 10, max_len), num_return_sequences=1)
                llm_out = result[0]['generated_text']
                
                # Enhanced repetition filter
                def remove_repeats(text):
                    words = text.split()
                    seen = set()
                    filtered = []
                    for i in range(len(words)):
                        # Check for repeated phrases of 2-4 words
                        for phrase_len in [4, 3, 2]:
                            if i >= phrase_len - 1:
                                phrase = ' '.join(words[i-phrase_len+1:i+1])
                                if phrase in seen:
                                    continue
                                seen.add(phrase)
                        filtered.append(words[i])
                    return ' '.join(filtered)
                prompt = remove_repeats(llm_out)
            except Exception:
                pass
                
        return (prompt,)

    node = type(
        class_name,
        (object,),
        {
            "INPUT_TYPES": create_input_types,
            "RETURN_TYPES": ("STRING",),
            "RETURN_NAMES": ("prompt",),
            "FUNCTION": "generate_prompt",
            "CATEGORY": "\U0001F310 Globetrotter",
            "generate_prompt": generate_prompt,
        }
    )
    return class_name, node


for country_data in COUNTRIES:
    c_name, c_node = create_attire_node(country_data)
    NODE_CLASS_MAPPINGS[c_name] = c_node
    NODE_DISPLAY_NAME_MAPPINGS[c_name] = f"{country_data['name']} Attire"
