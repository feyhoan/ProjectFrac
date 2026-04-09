"""
renderer.py - Logic for converting data arrays into images.

This module handles:
- Converting fractal data arrays to RGB images using Pillow/PIL
- Applying color palettes (Classic, Fire, Ocean)
- Generating placeholder images for the main menu
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, List, Optional


# Color palettes mapping iteration values to RGB colors
# Each palette is a list of (position, R, G, B) tuples where position is 0.0 to 1.0
COLOR_PALETTES = {
    "Classic": [
        (0.0, 0, 0, 0),
        (0.2, 0, 0, 50),
        (0.4, 0, 50, 100),
        (0.6, 50, 100, 150),
        (0.8, 150, 200, 255),
        (1.0, 255, 255, 255)
    ],
    "Fire": [
        (0.0, 0, 0, 0),
        (0.2, 50, 0, 0),
        (0.4, 150, 50, 0),
        (0.6, 200, 100, 0),
        (0.8, 255, 200, 50),
        (1.0, 255, 255, 200)
    ],
    "Ocean": [
        (0.0, 0, 0, 20),
        (0.2, 0, 20, 50),
        (0.4, 0, 50, 100),
        (0.6, 0, 100, 150),
        (0.8, 50, 150, 200),
        (1.0, 150, 200, 255)
    ],
    "Forest": [
        (0.0, 0, 0, 0),
        (0.2, 0, 30, 0),
        (0.4, 0, 80, 20),
        (0.6, 20, 120, 50),
        (0.8, 50, 180, 80),
        (1.0, 100, 220, 120)
    ],
    "Purple Haze": [
        (0.0, 0, 0, 0),
        (0.2, 30, 0, 50),
        (0.4, 80, 20, 120),
        (0.6, 120, 50, 180),
        (0.8, 180, 100, 220),
        (1.0, 220, 180, 255)
    ]
}


def interpolate_color(
    t: float, 
    palette: List[Tuple[float, int, int, int]]
) -> Tuple[int, int, int]:
    """
    Interpolate between colors in a palette based on a normalized value.
    
    Args:
        t: Normalized value between 0.0 and 1.0.
        palette: List of (position, R, G, B) tuples sorted by position.
    
    Returns:
        A tuple of (R, G, B) values.
    """
    # Clamp t to valid range
    t = max(0.0, min(1.0, t))
    
    # Find the two palette entries to interpolate between
    for i in range(len(palette) - 1):
        pos1, r1, g1, b1 = palette[i]
        pos2, r2, g2, b2 = palette[i + 1]
        
        if pos1 <= t <= pos2:
            # Calculate interpolation factor
            if pos2 == pos1:
                factor = 0.0
            else:
                factor = (t - pos1) / (pos2 - pos1)
            
            # Interpolate each channel
            r = int(r1 + factor * (r2 - r1))
            g = int(g1 + factor * (g2 - g1))
            b = int(b1 + factor * (b2 - b1))
            
            return (r, g, b)
    
    # Fallback: return last color
    _, r, g, b = palette[-1]
    return (r, g, b)


def apply_colormap(
    data: np.ndarray, 
    palette_name: str = "Classic",
    normalize: bool = True
) -> np.ndarray:
    """
    Apply a color palette to fractal data.
    
    Args:
        data: 2D numpy array with iteration counts or similar values.
        palette_name: Name of the color palette to use.
        normalize: If True, normalize data to 0-1 range based on max value.
    
    Returns:
        A 3D numpy array of shape (height, width, 3) with RGB values.
    """
    # Get the palette
    palette = COLOR_PALETTES.get(palette_name, COLOR_PALETTES["Classic"])
    
    # Normalize data if needed
    if normalize:
        max_val = np.max(data)
        if max_val > 0:
            normalized_data = data / max_val
        else:
            normalized_data = data
    else:
        normalized_data = data
    
    # Create output array
    height, width = data.shape
    result = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Apply color mapping to each pixel
    for y in range(height):
        for x in range(width):
            t = normalized_data[y, x]
            r, g, b = interpolate_color(t, palette)
            result[y, x] = [r, g, b]
    
    return result


def data_to_image(
    data: np.ndarray,
    palette_name: str = "Classic",
    background_color: Optional[Tuple[int, int, int]] = None
) -> Image.Image:
    """
    Convert fractal data to a PIL Image.
    
    Args:
        data: 2D numpy array with fractal data.
        palette_name: Name of the color palette to use.
        background_color: Optional RGB tuple for background color.
                         If None, uses the palette's first color for zero values.
    
    Returns:
        A PIL Image object.
    """
    # Apply colormap
    rgb_data = apply_colormap(data, palette_name)
    
    # Handle background color for zero values
    if background_color is not None:
        mask = data == 0
        rgb_data[mask] = background_color
    
    # Create PIL image
    image = Image.fromarray(rgb_data, mode='RGB')
    
    return image


def create_placeholder_image(
    text: str,
    width: int = 200,
    height: int = 120,
    bg_color: Tuple[int, int, int] = (60, 60, 70),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Create a placeholder image with text for the main menu cards.
    
    Args:
        text: Text to display on the placeholder.
        width: Width of the image in pixels.
        height: Height of the image in pixels.
        bg_color: Background color as RGB tuple.
        text_color: Text color as RGB tuple.
    
    Returns:
        A PIL Image object.
    """
    # Create image with solid background
    image = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fall back to bitmap font if not available
    try:
        # Try common system fonts
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            # Fall back to default bitmap font
            font = ImageFont.load_default()
    
    # Get text bounding box and center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill=text_color, font=font)
    
    # Add a decorative border
    draw.rectangle([0, 0, width - 1, height - 1], outline=(100, 100, 120), width=2)
    
    return image


def get_available_palettes() -> List[str]:
    """
    Get a list of available color palette names.
    
    Returns:
        List of palette name strings.
    """
    return list(COLOR_PALETTES.keys())


def render_fractal(
    fractal_type: str,
    params: dict,
    width: int,
    height: int,
    palette_name: str = "Classic",
    background_color: Optional[Tuple[int, int, int]] = None
) -> Image.Image:
    """
    Complete rendering pipeline: generate fractal data and convert to image.
    
    This is the main entry point for rendering fractals.
    
    Args:
        fractal_type: Type of fractal ("mandelbrot", "julia", or "tree").
        params: Dictionary of parameters for the fractal.
        width: Width of the output image in pixels.
        height: Height of the output image in pixels.
        palette_name: Name of the color palette to use.
        background_color: Optional RGB tuple for background color.
    
    Returns:
        A PIL Image object containing the rendered fractal.
    """
    # Import here to avoid circular imports
    from core import generate_fractal
    
    # Generate the fractal data
    data = generate_fractal(fractal_type, width, height, params)
    
    # Convert to image
    image = data_to_image(data, palette_name, background_color)
    
    return image
