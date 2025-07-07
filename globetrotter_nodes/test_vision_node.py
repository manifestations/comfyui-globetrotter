#!/usr/bin/env python3
"""
Simple test script for the Ollama Vision Node.
This script tests the basic functionality without ComfyUI.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from ollama_vision_node import OllamaVisionNode
import torch
import numpy as np

def test_vision_node():
    """Test the basic functionality of the vision node."""
    print("Testing Ollama Vision Node...")
    
    # Test model fetching
    print("\n1. Testing model fetching...")
    models = OllamaVisionNode.get_ollama_vision_models("http://127.0.0.1:11434/api/generate")
    print(f"Found {len(models)} vision models: {models}")
    
    # Test description prompts
    print("\n2. Testing description prompts...")
    prompts = OllamaVisionNode.get_description_prompts()
    print(f"Available description styles: {list(prompts.keys())}")
    
    # Test INPUT_TYPES
    print("\n3. Testing INPUT_TYPES...")
    try:
        inputs = OllamaVisionNode.INPUT_TYPES()
        print("INPUT_TYPES loaded successfully")
        print(f"Required inputs: {list(inputs['required'].keys())}")
        print(f"Optional inputs: {list(inputs['optional'].keys())}")
    except Exception as e:
        print(f"Error loading INPUT_TYPES: {e}")
    
    # Test tensor to base64 conversion
    print("\n4. Testing tensor conversion...")
    try:
        # Create a simple test image tensor (3x64x64 RGB)
        test_tensor = torch.rand(3, 64, 64)
        base64_result = OllamaVisionNode.tensor_to_base64(test_tensor)
        if base64_result:
            print(f"Successfully converted tensor to base64 (length: {len(base64_result)})")
        else:
            print("Failed to convert tensor to base64")
    except Exception as e:
        print(f"Error in tensor conversion: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_vision_node()
