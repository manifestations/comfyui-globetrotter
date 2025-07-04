# ComfyUI Globetrotter Nodes

A collection of custom ComfyUI nodes and utilities for generating AI image prompts representing the diverse attire, cultures, regions, and appearances of the world. This project is designed for easy extension to new countries, cultures, and body parts, using a modular JSON-based data structure and dynamic node generation.

## Features
- **Dynamic Node Generation**: Automatically creates a node for each country based on its JSON configuration.
- **Country-Specific Attire Options**: Each node provides dropdowns for attire options specific to the country and body part, as defined in the JSON files.
- **Regional Appearance Support**: Nodes include a region dropdown (from `appearance/<country>.json`) and always add region-specific skin tone and hair to the prompt.
- **Default Hair Style Dropdown**: Every node includes a hair style dropdown (from `attire/head/hair/generic.json`), with fallback to the region's default hair if not selected.
- **Extensible JSON Database**: Add new countries, body parts, or attire/appearance options by simply updating or adding JSON files.
- **Robust JSON Loading**: Only valid JSON files with a `body_part` entry are used for dropdown inputs.
- **Utility Nodes**: Includes text combiner and LLM prompt nodes for advanced workflows.
- **Expanded multicultural and regional attire/appearance libraries for India and Indonesia**, including Christian, Parsee, Muslim, tribal, North East, Kashmiri, Sulawesi, Bali, Java, and more.
- **Pose Support:** Country-specific pose dropdowns (e.g., for Indonesia).
- **Attire Descriptions:** Prompts now include both attire name and a short description.
- **Experimental LLM-based Prompt Rewriting:** Optionally rewrite prompts using a local Hugging Face `distilgpt2` model (toggle in node UI). Includes repetition filtering and output length control.

## Directory Structure
```
attire/                # JSON attire database (organized by body part and country)
appearance/            # JSON appearance/region database (by country)
globetrotter_nodes/    # All custom node and utility code
  dynamic_nodes.py     # Main dynamic node generation logic
  ollama_llm_node.py   # LLM-based prompt generation node
  text_combiner_node.py # Utility node for combining text inputs
data/                  # Art styles, cameras, lighting, etc.
tests/                 # Unit tests
README.md              # Project documentation
requirements.txt       # Python dependencies
```

## Installation
1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```sh
   git clone <repository-url> <ComfyUI-path>/custom_nodes/comfyui-globetrotter
   ```
2. (Optional) Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Restart ComfyUI to load the new nodes.

## Usage
### Dynamic Attire & Appearance Nodes
- Each country-specific node is generated based on the JSON files in the `attire/` and `appearance/` directories.
- Dropdown inputs are created for:
  - **Region**: Populated from `appearance/<country>.json` (e.g., Java, Papua, Sulawesi North, etc.).
  - **Hair Style**: Populated from `attire/head/hair/generic.json` (e.g., "bun", "ponytail", etc.).
  - **Body Parts**: As defined in the JSON files (e.g., "Blangkon", "Siger").
  - "none" and "random" are default options for attire and hair.
- The node generates a text prompt based on the selected options, which can be used in AI workflows.
- If no hair style is selected, the region's default hair is used. The region's skin tone and hair are always included in the prompt.

#### Example Appearance JSON (appearance/id.json):
```json
{
  "country": "id",
  "regions": [
    {
      "name": "Java",
      "description": "Medium to dark skin, straight or wavy black hair, Javanese features.",
      "skin_tone": "medium to dark",
      "hair": "black, straight or wavy"
    },
    {
      "name": "Sulawesi North",
      "description": "Light to medium skin, straight black hair, Minahasan and other North Sulawesi features.",
      "skin_tone": "light to medium",
      "hair": "black, straight"
    }
    // ... more regions ...
  ]
}
```

#### Example Hair Styles JSON (attire/head/hair/generic.json):
```json
[
  "long braid",
  "bun",
  "ponytail",
  "buzz cut",
  "random"
]
```

### Adding New Countries, Attire, or Appearance
1. Create a new JSON file in the appropriate directory under `attire/` or `appearance/`.
2. For attire, include:
   - `country`: The country code (e.g., "id" for Indonesia).
   - `body_part`: The body part this attire applies to (e.g., "head").
   - `attires`: A list of attire objects, each with at least a `name` field.
3. For appearance, include:
   - `country`: The country code.
   - `regions`: A list of region objects, each with `name`, `description`, `skin_tone`, and `hair` fields.
4. Restart ComfyUI to load the new node.

### Utility Nodes
- **Text Combiner Node**: Combines multiple text inputs into a single string.
- **LLM Prompt Node**: Generates prompts using a language model with additional dropdown options (e.g., camera settings).

### Experimental Features

- **LLM-based Prompt Rewriting:**  
  Each attire node includes an `experimental_llm_rewrite` toggle. When enabled, the generated prompt is rewritten using a local Hugging Face `distilgpt2` model (requires `transformers` in `requirements.txt`).  
  - Output is filtered to reduce repetition and capped in length.
  - If the model or dependencies are missing, the feature is silently skipped.
  - This is an experimental feature and may produce variable results.

### Windows Launch and Packaging

- A `launch.bat` file is provided for easy launching with venv activation.
- Instructions for creating a Windows shortcut and pinning to the taskbar are included in the repository.
- The project is ready for GitHub release and versioning.

## Contributing
- Follow the existing structure and naming conventions.
- Add tests for new utilities or nodes in the `tests/` directory.
- Ensure JSON files are valid and include the required fields.

## Requirements
- All required dependencies (including `transformers` and `pyyaml`) are listed in `requirements.txt`.  
  To use the LLM-based prompt rewriting, ensure you have installed all dependencies:
  ```sh
  pip install -r requirements.txt
  ```
