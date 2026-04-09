"""
renderer.py - Логика для преобразования массивов данных в изображения.

Этот модуль содержит функции для:
- Применения цветовых палитр к данным фракталов
- Создания изображений PIL из numpy массивов
- Генерации предварительных изображений для меню
- Композитинга с цветом фона
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, Any, Optional, List
import hashlib


# Цветовые палитры (список кортежей RGB)
COLOR_PALETTES = {
    "Классическая": [
        (0, 0, 0),      # Черный (для множества Мандельброта)
        (25, 25, 112),  # Темно-синий
        (0, 128, 255),  # Голубой
        (0, 255, 255),  # Циан
        (255, 255, 0),  # Желтый
        (255, 165, 0),  # Оранжевый
        (255, 0, 0),    # Красный
        (255, 255, 255) # Белый
    ],
    "Огонь": [
        (0, 0, 0),
        (139, 0, 0),    # Темно-красный
        (255, 69, 0),   # Огненно-красный
        (255, 140, 0),  # Оранжевый
        (255, 215, 0),  # Золотой
        (255, 255, 0),  # Желтый
        (255, 255, 255) # Белый
    ],
    "Океан": [
        (0, 0, 0),
        (0, 0, 50),     # Очень темный синий
        (0, 0, 100),    # Темный синий
        (0, 100, 150),  # Синий
        (0, 150, 200),  # Светло-синий
        (0, 200, 255),  # Голубой
        (100, 255, 255) # Светлый циан
    ],
    "Лес": [
        (0, 0, 0),
        (0, 50, 0),     # Темно-зеленый
        (0, 100, 0),    # Зеленый
        (0, 150, 50),   # Средний зеленый
        (50, 200, 50),  # Светло-зеленый
        (100, 255, 100) # Яркий зеленый
    ],
    "Фиолетовая дымка": [
        (0, 0, 0),
        (75, 0, 75),    # Темно-фиолетовый
        (128, 0, 128),  # Фиолетовый
        (180, 50, 180), # Светло-фиолетовый
        (220, 100, 220) # Розово-фиолетовый
    ]
}


def get_available_palettes() -> List[str]:
    """
    Get a list of available color palette names.

    Returns:
        List of palette name strings.
    """
    return list(COLOR_PALETTES.keys())


def interpolate_color(color1: Tuple[int, int, int], color2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Интерполирует между двумя цветами."""
    t = max(0.0, min(1.0, t))
    r = int(color1[0] + (color2[0] - color1[0]) * t)
    g = int(color1[1] + (color2[1] - color1[1]) * t)
    b = int(color1[2] + (color2[2] - color1[2]) * t)
    return (r, g, b)


def apply_colormap(data: np.ndarray, palette_name: str = "Классическая") -> np.ndarray:
    """Применяет цветовую палитру к данным фрактала."""
    palette = COLOR_PALETTES.get(palette_name, COLOR_PALETTES["Классическая"])
    height, width = data.shape
    result = np.zeros((height, width, 3), dtype=np.uint8)
    
    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val == min_val:
        color = palette[len(palette) // 2]
        result[:, :] = color
        return result
    
    num_segments = len(palette) - 1
    norm_data = (data - min_val) / (max_val - min_val) if max_val > min_val else data
    
    for i in range(height):
        for j in range(width):
            norm_val = norm_data[i, j]
            palette_idx = norm_val * num_segments
            idx_low = int(palette_idx)
            idx_high = min(idx_low + 1, len(palette) - 1)
            t = palette_idx - idx_low
            color = interpolate_color(palette[idx_low], palette[idx_high], t)
            result[i, j] = color
    
    return result


def data_to_image(
    data: np.ndarray,
    palette_name: str = "Классическая",
    background_color: Tuple[int, int, int] = None,
    fractal_type: str = "mandelbrot"
) -> Image.Image:
    """Преобразует массив данных в изображение PIL."""
    if background_color is None:
        background_color = (0, 0, 0)
    
    rgb_data = apply_colormap(data, palette_name)
    
    if fractal_type == "tree":
        alpha = np.zeros(data.shape, dtype=np.uint8)
        mask = data > 0
        alpha[mask] = 255
        
        rgba_data = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
        rgba_data[:, :, :3] = rgb_data
        rgba_data[:, :, 3] = alpha
        
        img = Image.fromarray(rgba_data, mode='RGBA')
        background = Image.new('RGBA', img.size, background_color + (255,))
        background.paste(img, (0, 0), img)
        return background.convert('RGB')
    else:
        img = Image.fromarray(rgb_data, mode='RGB')
        
        if background_color != (0, 0, 0):
            bg_img = Image.new('RGB', img.size, background_color)
            data_array = np.array(img)
            not_black = np.any(data_array > 10, axis=2)
            mask = Image.fromarray(not_black.astype(np.uint8) * 255, mode='L')
            bg_img.paste(img, (0, 0), mask)
            img = bg_img
        
        return img


def create_preview_image(fractal_type: str, params: Dict[str, Any], width: int = 200, height: int = 150) -> Image.Image:
    """Создает предварительное изображение для карточки фрактала."""
    from core import generate_fractal
    
    render_params = params.copy()
    if "max_iterations" in render_params:
        render_params["max_iterations"] = min(render_params["max_iterations"], 50)
    if "depth" in render_params:
        render_params["depth"] = min(render_params.get("depth", 9), 8)
    
    try:
        data = generate_fractal(fractal_type, width, height, render_params, use_cache=False)
        palette_map = {"mandelbrot": "Океан", "julia": "Огонь", "tree": "Лес"}
        palette = palette_map.get(fractal_type, "Классическая")
        img = data_to_image(data, palette_name=palette, background_color=(30, 30, 40), fractal_type=fractal_type)
        return img
    except Exception as e:
        print(f"Ошибка при создании превью: {e}")
        return create_placeholder_image(fractal_type, width, height)


def create_placeholder_image(text: str = "Fractal", width: int = 260, height: int = 180,
                             bg_color: tuple = (50, 50, 80), text_color: tuple = (255, 255, 255)) -> Image.Image:
    """Создаёт заглушку-превью для карточки фрактала."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Пытаемся использовать системный шрифт, иначе fallback на стандартный
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # Центрируем текст
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) / 2
    y = (height - text_h) / 2

    draw.text((x, y), text, fill=text_color, font=font)
    return img


# Кэш для рендеринга
_render_cache = {}
_fractal_data_cache = {}


def get_cached_image(cache_key: str) -> Optional[Image.Image]:
    return _render_cache.get(cache_key)


def cache_image(cache_key: str, img: Image.Image):
    _render_cache[cache_key] = img


def get_cached_fractal_data(cache_key: str) -> Optional[np.ndarray]:
    return _fractal_data_cache.get(cache_key)


def cache_fractal_data(cache_key: str, data: np.ndarray):
    _fractal_data_cache[cache_key] = data


def clear_all_caches():
    _render_cache.clear()
    _fractal_data_cache.clear()


def generate_cache_key(fractal_type: str, params: Dict[str, Any], width: int, height: int) -> str:
    param_str = "_".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                         for k, v in sorted(params.items()))
    base_str = f"{fractal_type}_{width}x{height}_{param_str}"
    return hashlib.md5(base_str.encode()).hexdigest()[:16]


def render_fractal(
    fractal_type: str,
    params: Dict[str, Any],
    width: int = 800,
    height: int = 600,
    palette_name: str = "Классическая",
    background_color: Tuple[int, int, int] = None,
    use_cache: bool = True,
    cached_fractal_data: np.ndarray = None
) -> Tuple[Image.Image, np.ndarray]:
    """Полный цикл рендеринга фрактала."""
    from core import generate_fractal
    
    if background_color is None:
        background_color = (0, 0, 0)
    
    cache_key = generate_cache_key(fractal_type, params, width, height)
    data_cache_key = f"data_{cache_key}"
    
    if cached_fractal_data is not None:
        data = cached_fractal_data
    elif use_cache:
        data = get_cached_fractal_data(data_cache_key)
        if data is None:
            data = generate_fractal(fractal_type, width, height, params, use_cache=True)
            cache_fractal_data(data_cache_key, data)
    else:
        data = generate_fractal(fractal_type, width, height, params, use_cache=False)
    
    img_cache_key = f"{cache_key}_{palette_name}_{background_color}"
    if use_cache:
        cached_img = get_cached_image(img_cache_key)
        if cached_img is not None:
            return cached_img, data
    
    img = data_to_image(data, palette_name, background_color, fractal_type)
    
    if use_cache:
        cache_image(img_cache_key, img)
    
    return img, data
