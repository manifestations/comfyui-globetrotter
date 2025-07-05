# Globetrotter Premium Controls Guide

This guide explains all the premium controls available in the Globetrotter nodes and how to use them for exceptional results.

## Premium Controls Overview

### Subject Definition
- **Age**: Character age (refined options for sophisticated output)
- **Gender**: Character gender (Female, Male, Non-binary, Unspecified)
- **Region**: Specific regional appearance within the country

### Aesthetic Characteristics
- **Hair Style**: Elegant hair styling options
- **Emotion**: Sophisticated emotional expressions (confident, serene, elegant, graceful, etc.)
- **Attire Controls** (Body Parts): Premium clothing options for different body parts
  - Head, Upper Body, Legs, Waist, etc.
  - Gender-appropriate filtering with premium quality options
  - Each body part shows refined clothing options for the selected country

### Composition and Context
- **Pose**: Elegant, professional poses and gestures
- **Cultural Element**: Refined cultural activities and traditions

### Environment and Atmosphere
- **Setting**: Premium locations and sophisticated environments
- **Atmosphere**: Elegant lighting and refined atmospheric conditions

### Technical Excellence
- **Detail Level**: Premium quality levels
  - `premium`: Enhanced quality with professional standards
  - `masterpiece`: Museum-quality excellence
  - `cinematic`: Professional cinematography standards
  - `photorealistic`: Ultra-high definition realism
  - `artistic`: Fine art mastery

- **Composition**: Professional composition styles
  - `rule of thirds`: Classic professional photography
  - `portrait`: Portrait-focused professional framing
  - `close-up`: Intimate, detailed shots
  - `full body`: Complete, elegant presentation
  - `centered`: Balanced, sophisticated composition
  - `wide shot`: Environmental context with style
  - `environmental`: Setting-focused premium composition

- **Prompt Optimization**: Premium prompt structuring
  - `premium`: Optimized for highest quality AI generation
  - `artistic`: Fine art focused optimization
  - `professional`: Commercial-grade professional output
  - `creative`: Maximum artistic expression
  - `technical`: Precision-focused technical excellence

## Premium Randomization System

### Randomization Levels
- **Off**: Consistent, premium results every time
- **Subtle**: Minimal, refined variations maintaining quality
- **Balanced**: Optimal creativity with premium standards
- **Creative**: Maximum artistic freedom with sophisticated output

### Seed Control
- **Seed**: Premium reproducibility
- Ensures consistent high-quality results
- Professional workflow compatibility

## Universal Options

Every dropdown includes these options:
- **none**: Skip this element (don't include in prompt)
- **random**: Let the system choose randomly from available options

## Advanced Controls

### LoRA Integration
- **LoRA Trigger**: Character name or trigger word for LoRA models
- Example: "john_doe" or "my_character"

### Custom Prompts
- **Custom Prompt**: Additional text to include in the final prompt
- Use for specific details not covered by other controls

### Experimental Features
- **Experimental LLM Rewrite**: AI-powered prompt enhancement
- Requires `transformers` library installed
- May improve prompt quality but adds processing time

## Usage Tips

### For Consistent Results
1. Set **Randomization** to "off"
2. Use specific choices instead of "random"
3. Keep the same **Seed** value

### For Creative Exploration
1. Set **Randomization** to "moderate" or "full"
2. Use "random" for elements you want to vary
3. Change the **Seed** for different variations

### For High Quality Output
1. Set **Detail Level** to "detailed" or higher
2. Use **Prompt Optimization** set to "ai_friendly"
3. Choose appropriate **Composition** style

### Gender-Aware Attire
- Attire options are automatically filtered based on selected gender
- Each clothing item has gender tags (male, female, unisex)
- Inappropriate combinations are automatically skipped

## Best Practices

1. **Start Simple**: Begin with basic settings and gradually add complexity
2. **Use Presets**: Save successful combinations for reuse
3. **Test Systematically**: Change one parameter at a time to understand effects
4. **Match Style**: Ensure all elements (detail level, composition, optimization) work together
5. **Cultural Sensitivity**: Respect cultural contexts when combining elements

## Troubleshooting

### Empty Outputs
- Check that country data files exist in the `data/` directory
- Verify at least one attire option is selected
- Ensure gender-appropriate attire is available

### Inconsistent Results
- Use "off" randomization for consistent outputs
- Check that the same seed is being used
- Verify all input parameters are identical

### Poor Quality
- Increase detail level
- Use "ai_friendly" prompt optimization
- Add specific composition guidelines
- Consider using custom prompts for specific requirements
