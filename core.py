"""
core.py - Mathematical logic and algorithms for generating fractal data.

This module contains the core mathematical implementations for:
- Mandelbrot Set
- Julia Set
- Fractal Tree

Optimized with NumPy vectorization for faster rendering.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def mandelbrot_set(
    width: int,
    height: int,
    center_x: float = -0.5,
    center_y: float = 0.0,
    zoom: float = 1.0,
    max_iterations: int = 100
) -> np.ndarray:
    """
    Generate the Mandelbrot set fractal using vectorized NumPy operations.
    
    The Mandelbrot set is the set of complex numbers c for which the function
    f(z) = z² + c does not diverge when iterated from z = 0.
    
    Args:
        width: Width of the output image in pixels.
        height: Height of the output image in pixels.
        center_x: X-coordinate of the center of the view.
        center_y: Y-coordinate of the center of the view.
        zoom: Zoom level (higher = more zoomed in).
        max_iterations: Maximum number of iterations to determine divergence.
                       Higher values give more detail but are slower.
    
    Returns:
        A 2D numpy array where each value represents the number of iterations
        before divergence (or max_iterations if it didn't diverge).
    """
    # Calculate the bounds based on center and zoom
    x_min = center_x - 2.0 / zoom
    x_max = center_x + 2.0 / zoom
    y_min = center_y - 2.0 / zoom
    y_max = center_y + 2.0 / zoom
    
    # Create coordinate grids
    x = np.linspace(x_min, x_max, width, dtype=np.float64)
    y = np.linspace(y_min, y_max, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    
    # Initialize complex plane
    C = X + 1j * Y
    Z = np.zeros_like(C)
    
    # Iteration count array
    iterations = np.zeros(Z.shape, dtype=np.int32)
    
    # Track which points haven't diverged yet
    not_diverged = np.ones(Z.shape, dtype=bool)
    
    # Vectorized iteration loop
    for i in range(max_iterations):
        Z[not_diverged] = Z[not_diverged] ** 2 + C[not_diverged]
        
        # Check for divergence (|z| > 2)
        diverged = np.abs(Z) > 2
        newly_diverged = diverged & not_diverged
        iterations[newly_diverged] = i
        not_diverged &= ~diverged
        
        # Early exit if all points have diverged
        if not np.any(not_diverged):
            break
    
    # Points that never diverged get max_iterations
    iterations[not_diverged] = max_iterations
    
    return iterations


def julia_set(
    width: int,
    height: int,
    c_real: float = -0.7,
    c_imag: float = 0.27017,
    center_x: float = 0.0,
    center_y: float = 0.0,
    zoom: float = 1.0,
    max_iterations: int = 100
) -> np.ndarray:
    """
    Generate the Julia set fractal using vectorized NumPy operations.
    
    The Julia set is similar to the Mandelbrot set but uses a fixed complex
    constant c and varies the initial z value. Different values of c produce
    dramatically different fractals.
    
    Args:
        width: Width of the output image in pixels.
        height: Height of the output image in pixels.
        c_real: Real part of the complex constant c.
        c_imag: Imaginary part of the complex constant c.
        center_x: X-coordinate of the center of the view.
        center_y: Y-coordinate of the center of the view.
        zoom: Zoom level (higher = more zoomed in).
        max_iterations: Maximum number of iterations to determine divergence.
                       Higher values give more detail but are slower.
    
    Returns:
        A 2D numpy array where each value represents the number of iterations
        before divergence (or max_iterations if it didn't diverge).
    """
    # Calculate the bounds based on center and zoom
    x_min = center_x - 2.0 / zoom
    x_max = center_x + 2.0 / zoom
    y_min = center_y - 2.0 / zoom
    y_max = center_y + 2.0 / zoom
    
    # Create coordinate grids
    x = np.linspace(x_min, x_max, width, dtype=np.float64)
    y = np.linspace(y_min, y_max, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    
    # Initialize complex plane with the starting z values
    Z = X + 1j * Y
    
    # The fixed complex constant c
    C = c_real + 1j * c_imag
    
    # Iteration count array
    iterations = np.zeros(Z.shape, dtype=np.int32)
    
    # Track which points haven't diverged yet
    not_diverged = np.ones(Z.shape, dtype=bool)
    
    # Vectorized iteration loop
    for i in range(max_iterations):
        Z[not_diverged] = Z[not_diverged] ** 2 + C
        
        # Check for divergence (|z| > 2)
        diverged = np.abs(Z) > 2
        newly_diverged = diverged & not_diverged
        iterations[newly_diverged] = i
        not_diverged &= ~diverged
        
        # Early exit if all points have diverged
        if not np.any(not_diverged):
            break
    
    # Points that never diverged get max_iterations
    iterations[not_diverged] = max_iterations
    
    return iterations


def fractal_tree(
    width: int,
    height: int,
    branch_angle: float = 0.5,
    length_ratio: float = 0.7,
    max_depth: int = 10,
    start_length: float = 150.0
) -> np.ndarray:
    """
    Generate a fractal tree using iterative branching (optimized).
    
    This creates a binary tree where each branch splits into two smaller
    branches at a specified angle. The tree is rendered as a distance field
    for smooth coloring.
    
    Uses an iterative stack-based approach instead of recursion for better
    performance.
    
    Args:
        width: Width of the output image in pixels.
        height: Height of the output image in pixels.
        branch_angle: Angle between branches in radians.
                     Typical range: 0.3 to 1.0 radians.
        length_ratio: Ratio of child branch length to parent branch length.
                     Typical range: 0.5 to 0.8.
        max_depth: Maximum recursion depth. Higher = more branches but slower.
                  Typical range: 8 to 12.
        start_length: Initial trunk length in pixels.
    
    Returns:
        A 2D numpy array representing the tree structure.
        Values represent the depth at which each pixel was reached,
        or 0 if no branch reached that pixel.
    """
    # Initialize output array
    result = np.zeros((height, width), dtype=np.float64)
    
    # Starting position (bottom center)
    start_x = width / 2.0
    start_y = float(height - 50)  # Leave some margin at bottom
    
    # Use iterative stack-based approach (more efficient than recursion)
    # Stack entries: (x, y, angle, length, depth)
    stack = [(start_x, start_y, 0.0, start_length, max_depth)]
    
    while stack:
        x, y, angle, length, depth = stack.pop()
        
        if depth == 0 or length < 1:
            continue
        
        # Calculate end point of this branch
        end_x = x + length * np.sin(angle)
        end_y = y - length * np.cos(angle)  # Negative because y goes down
        
        # Draw the branch using vectorized line drawing
        num_points = max(int(length), 1)
        t_values = np.linspace(0, 1, num_points)
        
        px_values = (x + t_values * (end_x - x)).astype(int)
        py_values = (y + t_values * (end_y - y)).astype(int)
        
        # Filter points within bounds
        valid_mask = (px_values >= 0) & (px_values < width) & \
                     (py_values >= 0) & (py_values < height)
        
        if np.any(valid_mask):
            normalized_depth = depth / max_depth
            for px, py in zip(px_values[valid_mask], py_values[valid_mask]):
                if normalized_depth > result[py, px]:
                    result[py, px] = normalized_depth
        
        # Add child branches to stack
        new_length = length * length_ratio
        
        # Right branch
        stack.append((end_x, end_y, angle + branch_angle, new_length, depth - 1))
        # Left branch
        stack.append((end_x, end_y, angle - branch_angle, new_length, depth - 1))
    
    return result


def get_fractal_parameters(fractal_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get the parameter definitions for a specific fractal type.
    
    This is used by the UI to dynamically generate controls.
    
    Args:
        fractal_type: One of "mandelbrot", "julia", or "tree".
    
    Returns:
        A dictionary mapping parameter names to their configurations.
        Each configuration includes:
        - type: "scale" or "entry"
        - min: Minimum value (for scales)
        - max: Maximum value (for scales)
        - default: Default value
        - step: Step size for scales
        - description: Human-readable description of what the parameter does
    """
    parameters = {
        "mandelbrot": {
            "center_x": {
                "type": "entry",
                "default": -0.5,
                "description": "X-coordinate of view center. Pan left/right."
            },
            "center_y": {
                "type": "entry",
                "default": 0.0,
                "description": "Y-coordinate of view center. Pan up/down."
            },
            "zoom": {
                "type": "scale",
                "min": 0.1,
                "max": 100.0,
                "default": 1.0,
                "step": 0.1,
                "description": "Zoom level. Higher = more zoomed in (more detail)."
            },
            "max_iterations": {
                "type": "scale",
                "min": 10,
                "max": 500,
                "default": 100,
                "step": 10,
                "description": "Iterations: Higher = more detail but slower rendering."
            }
        },
        "julia": {
            "c_real": {
                "type": "scale",
                "min": -2.0,
                "max": 2.0,
                "default": -0.7,
                "step": 0.01,
                "description": "Real part of c: Changes the fractal shape horizontally."
            },
            "c_imag": {
                "type": "scale",
                "min": -2.0,
                "max": 2.0,
                "default": 0.27017,
                "step": 0.01,
                "description": "Imaginary part of c: Changes the fractal shape vertically."
            },
            "center_x": {
                "type": "entry",
                "default": 0.0,
                "description": "X-coordinate of view center. Pan left/right."
            },
            "center_y": {
                "type": "entry",
                "default": 0.0,
                "description": "Y-coordinate of view center. Pan up/down."
            },
            "zoom": {
                "type": "scale",
                "min": 0.1,
                "max": 100.0,
                "default": 1.0,
                "step": 0.1,
                "description": "Zoom level. Higher = more zoomed in."
            },
            "max_iterations": {
                "type": "scale",
                "min": 10,
                "max": 500,
                "default": 100,
                "step": 10,
                "description": "Iterations: Higher = more detail but slower rendering."
            }
        },
        "tree": {
            "branch_angle": {
                "type": "scale",
                "min": 0.1,
                "max": 1.5,
                "default": 0.5,
                "step": 0.05,
                "description": "Angle between branches (radians). Wider = more spread out."
            },
            "length_ratio": {
                "type": "scale",
                "min": 0.5,
                "max": 0.9,
                "default": 0.7,
                "step": 0.01,
                "description": "Child branch length ratio. Higher = longer branches."
            },
            "max_depth": {
                "type": "scale",
                "min": 5,
                "max": 15,
                "default": 10,
                "step": 1,
                "description": "Recursion depth. Higher = more branches but much slower."
            },
            "start_length": {
                "type": "scale",
                "min": 50,
                "max": 300,
                "default": 150,
                "step": 10,
                "description": "Initial trunk length in pixels."
            }
        }
    }
    
    return parameters.get(fractal_type, {})


def generate_fractal(
    fractal_type: str,
    width: int,
    height: int,
    params: Dict[str, float]
) -> np.ndarray:
    """
    Generate a fractal based on the specified type and parameters.
    
    This is the main entry point for fractal generation.
    
    Args:
        fractal_type: One of "mandelbrot", "julia", or "tree".
        width: Width of the output image in pixels.
        height: Height of the output image in pixels.
        params: Dictionary of parameter values for the fractal.
    
    Returns:
        A 2D numpy array containing the fractal data.
    """
    if fractal_type == "mandelbrot":
        return mandelbrot_set(
            width=width,
            height=height,
            center_x=params.get("center_x", -0.5),
            center_y=params.get("center_y", 0.0),
            zoom=params.get("zoom", 1.0),
            max_iterations=int(params.get("max_iterations", 100))
        )
    
    elif fractal_type == "julia":
        return julia_set(
            width=width,
            height=height,
            c_real=params.get("c_real", -0.7),
            c_imag=params.get("c_imag", 0.27017),
            center_x=params.get("center_x", 0.0),
            center_y=params.get("center_y", 0.0),
            zoom=params.get("zoom", 1.0),
            max_iterations=int(params.get("max_iterations", 100))
        )
    
    elif fractal_type == "tree":
        return fractal_tree(
            width=width,
            height=height,
            branch_angle=params.get("branch_angle", 0.5),
            length_ratio=params.get("length_ratio", 0.7),
            max_depth=int(params.get("max_depth", 10)),
            start_length=params.get("start_length", 150.0)
        )
    
    else:
        raise ValueError(f"Unknown fractal type: {fractal_type}")
