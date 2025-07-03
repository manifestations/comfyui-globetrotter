class TextCombinerNode:
    """
    A node to combine multiple text inputs into a single comma-separated string.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", ),
                "text_2": ("STRING", ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_text",)
    FUNCTION = "combine_text"
    CATEGORY = "🌐 Globetrotter/Utils"
    NAME = "Text Combiner"

    def combine_text(self, text_1, text_2):
        """
        Combines the provided text inputs into a single comma-separated string.
        """
        text_parts = [text for text in [text_1, text_2] if text and text.strip()]
        return (", ".join(text_parts),)

NODE_CLASS_MAPPINGS = {
    "TextCombinerNode": TextCombinerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextCombinerNode": "Text Combiner"
}