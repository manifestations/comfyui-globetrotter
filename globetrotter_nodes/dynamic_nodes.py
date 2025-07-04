import os
import json
import random

# --- Constants ---
BASE_DIR = os.path.dirname(__file__)
COUNTRIES_FILE = os.path.join(BASE_DIR, '..', 'data', 'countries.json')
ATTIRE_DIR = os.path.join(BASE_DIR, '..', 'data', 'attire')
GENERIC_HAIR_FILE = os.path.join(ATTIRE_DIR, 'head', 'hair', 'generic.json')

# --- Helper Functions ---


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


def load_attire_options(country_code):
    """
    Loads attire options for a specific country by directly checking the
    `body_part` entry in the JSON file.
    """
    country_attire_options = {}

    # Iterate through all JSON files in the attire directory for the given
    # country
    for root, _, files in os.walk(ATTIRE_DIR):
        for file in files:
            if file == f"{country_code}.json":
                json_path = os.path.join(root, file)
                with open(json_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        body_part = data.get("body_part")
                        if body_part and "attires" in data:
                            options = ["none", "random"] + [
                                attire["name"]
                                for attire in data["attires"]
                            ]
                            # Capitalize each word in each option except
                            # 'none' and 'random'
                            options = [
                                opt
                                if opt in ["none", "random"]
                                else opt.title()
                                for opt in options
                            ]
                            country_attire_options[
                                body_part.title()
                            ] = options
                    except json.JSONDecodeError:
                        pass
    return country_attire_options


def load_appearance_options(country_code: str):
    """Load region appearance options for a country from
    data/appearance/<country>.json."""
    appearance_file = os.path.join(
        BASE_DIR, "..", "data", "appearance", f"{country_code}.json"
    )
    if not os.path.exists(appearance_file):
        return []
    try:
        with open(appearance_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Capitalize each word in each region name
            regions = data.get("regions", [])
            for region in regions:
                if "name" in region:
                    region["name"] = region["name"].title()
            return regions
    except Exception:
        return []


def load_generic_hair_styles():
    """Load generic hair styles from generic.json."""
    try:
        with open(GENERIC_HAIR_FILE, "r", encoding="utf-8") as f:
            options = ["none", "random"] + json.load(f)
            # Capitalize each word in each option except 'none' and 'random'
            options = [
                opt if opt in ["none", "random"] else opt.title()
                for opt in options
            ]
            return options
    except Exception:
        return ["none", "random"]


def load_pose_options(country_code: str):
    """Load pose options for a country from data/poses/<country>.json."""
    pose_file = os.path.join(
        BASE_DIR, "..", "data", "poses", f"{country_code}.json"
    )
    if not os.path.exists(pose_file):
        return ["none", "random"]
    try:
        with open(pose_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            poses = data.get("poses", [])
            # Capitalize each word in each pose except 'none' and 'random'
            return ["none", "random"] + [p.title() for p in poses]
    except Exception:
        return ["none", "random"]


def format_attire_list(attires: list[str]) -> str:
    """Format a list of attires in natural language (comma and 'and')."""
    if not attires:
        return ""
    if len(attires) == 1:
        return attires[0]
    return ", ".join(attires[:-1]) + " and " + attires[-1]

# --- Pre-load static data ---
try:
    with open(COUNTRIES_FILE, "r", encoding="utf-8") as f:
        COUNTRIES = json.load(f)
except Exception as e:
    print(f"Fatal Error: Could not load countries.json: {e}")
    COUNTRIES = []

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

    # Pre-load attire options for this specific country
    country_attire_options = load_attire_options(country_code)
    region_options = load_appearance_options(country_code)
    region_names = [r['name'] for r in region_options] if region_options else []
    generic_hair_styles = load_generic_hair_styles()
    pose_options = load_pose_options(country_code)

    @classmethod
    def input_types(cls):
        """Return the required input types for the node."""
        required = {
            "region": (region_names if region_names else ["Unspecified"],),
            "hair_style": (generic_hair_styles,),
            "pose": (pose_options,),
            "gender": (["Unspecified", "Male", "Female", "Non-binary"],),
            "age": (
                [
                    "unspecified",
                    "infant",
                    "toddler",
                    "young kid",
                    "older kid",
                    "teenager",
                    "young adult",
                    "adult",
                    "middle aged",
                    "elderly",
                ],
            ),
        }
        for body_part, options in country_attire_options.items():
            if options:  # Ensure options exist for the body part
                required[body_part] = (options,)
        # Add randomization controls and text boxes at the bottom
        required["random_mode"] = (
            ["random", "fixed", "increment", "decrement"],
            {"default": "random"},
        )
        required["seed"] = ("INT", {"default": 0, "min": 0, "max": 2**32 - 1})
        required["lora_trigger"] = (
            "STRING",
            {"multiline": False, "default": ""},
        )
        required["custom_prompt"] = (
            "STRING",
            {"multiline": True, "default": ""},
        )
        required["experimental_llm_rewrite"] = (['off', 'on'], {"default": "off"})
        return {"required": required}

    def generate_prompt(
        self,
        region: str,
        hair_style: str,
        pose: str,
        gender: str,
        age: str,
        random_mode: str,
        seed: int,
        lora_trigger: str,
        custom_prompt: str,
        experimental_llm_rewrite: str = 'off',
        **kwargs,
    ) -> tuple[str]:
        """Generate a natural language prompt for the selected options."""
        # Set defaults if unspecified
        if gender == "Unspecified":
            gender = "Female"
        if age == "unspecified":
            age = "young adult"
        prompt_parts = []

        # Set up randomization
        rng = random.Random()
        if random_mode == "fixed":
            rng.seed(seed)
        elif random_mode == "increment":
            rng.seed(seed + 1)
        elif random_mode == "decrement":
            rng.seed(max(0, seed - 1))
        # else: system random (no seed set)

        # Validate region selection
        region_info = next(
            (r for r in region_options if r["name"] == region), None
        )
        region_display = (
            region
            if region and region != "Unspecified" and region_info
            else None
        )

        # Add LORA trigger phrase or fallback
        if lora_trigger.strip():
            if region_display:
                prompt_parts.append(
                    f"A {age} {gender} called {lora_trigger.strip()} from "
                    f"{region_display}, {country_name}"
                )
            else:
                prompt_parts.append(
                    f"A {age} {gender} called {lora_trigger.strip()} from "
                    f"{country_name}"
                )
        else:
            if region_display:
                prompt_parts.append(
                    f"A {age} {gender} from {region_display}, {country_name}"
                )
            else:
                prompt_parts.append(f"A {age} {gender} from {country_name}")

        # Add region description, skin_tone, and hair
        if region_info:
            desc = region_info.get("description", "")
            if desc:
                prompt_parts.append(desc)
            prompt_parts.append(
                f"skin tone: {region_info.get('skin_tone','unspecified')}"
            )
        # Hair style: use dropdown if not none/random, else use region default
        if hair_style not in ["none", "random", "Unspecified", ""]:
            prompt_parts.append(f"hair: {hair_style}")
        elif region_info:
            prompt_parts.append(
                f"hair: {region_info.get('hair','unspecified')}"
            )

        # Collect all selected attires with descriptions
        selected_attires = []
        for body_part, options in country_attire_options.items():
            selected = kwargs.get(body_part)
            if selected == "random":
                choosable = [
                    opt for opt in options if opt not in ["none", "random"]
                ]
                if choosable:
                    selected = rng.choice(choosable)
                else:
                    selected = "none"
            if selected and selected != "none":
                # Fetch description from JSON
                attire_desc = None
                # Find the JSON file for this body part
                attire_file = os.path.join(ATTIRE_DIR, body_part.lower(), f"{country_code}.json")
                if os.path.exists(attire_file):
                    try:
                        with open(attire_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for attire in data.get("attires", []):
                                if attire["name"].lower() == selected.lower():
                                    attire_desc = attire.get("description")
                                    break
                    except Exception:
                        pass
                if attire_desc:
                    selected_attires.append(f"{selected}: {attire_desc}")
                else:
                    selected_attires.append(selected)
        if selected_attires:
            prompt_parts.append(
                "wearing " + format_attire_list(selected_attires)
            )
        else:
            prompt_parts.append("with no specific attire")

        # Add pose if selected
        if pose not in ["none", "random", "Unspecified", ""]:
            prompt_parts.append(f"pose: {pose}")

        # Add custom prompt if provided
        if custom_prompt.strip():
            prompt_parts.append(custom_prompt.strip())

        # Capitalize the first letter and filter empty parts
        prompt = ", ".join([part for part in prompt_parts if part])
        prompt = prompt[:1].upper() + prompt[1:] if prompt else prompt
        # Experimental LLM rewrite
        if experimental_llm_rewrite == 'on' and LLM_AVAILABLE and prompt:
            try:
                # Reduce max_length to minimize repetition
                result = llm_rewriter(prompt, max_length=min(len(prompt.split()) + 5, 50), num_return_sequences=1)
                llm_out = result[0]['generated_text']
                # Simple repetition filter: remove repeated phrases longer than 3 words
                def remove_repeats(text):
                    words = text.split()
                    seen = set()
                    filtered = []
                    for i in range(len(words)):
                        phrase = ' '.join(words[max(0, i-3):i+1])
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
            "INPUT_TYPES": input_types,
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
