# __init__.py for comfyui-globetrotter
# This imports and exposes all dynamically generated nodes for ComfyUI
from .globetrotter_nodes.dynamic_nodes import (
    NODE_CLASS_MAPPINGS as DYNAMIC_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as DYNAMIC_NODE_DISPLAY_NAME_MAPPINGS
)
from .globetrotter_nodes.ollama_llm_node import (
    NODE_CLASS_MAPPINGS as LLM_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LLM_NODE_DISPLAY_NAME_MAPPINGS
)
from .globetrotter_nodes.text_combiner_node import (
    NODE_CLASS_MAPPINGS as COMBINER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COMBINER_NODE_DISPLAY_NAME_MAPPINGS
)
from .globetrotter_nodes.ollama_vision_node import (
    NODE_CLASS_MAPPINGS as VISION_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VISION_NODE_DISPLAY_NAME_MAPPINGS
)

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(DYNAMIC_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(LLM_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(COMBINER_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(VISION_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(DYNAMIC_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LLM_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(COMBINER_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(VISION_NODE_DISPLAY_NAME_MAPPINGS)

print(f"Globetrotter: Loaded {len(NODE_CLASS_MAPPINGS)} node classes.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
