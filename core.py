"""
core.py - Математическая логика и алгоритмы для генерации данных фракталов.

Этот модуль содержит основные математические реализации для:
- Множества Мандельброта
- Множества Жюлиа
- Фрактального дерева

Оптимизировано с использованием векторизации NumPy и JIT-компиляции Numba.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from functools import lru_cache

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


NUMBA_AVAILABLE = False  # Отключаем numba для совместимости, используем чистую векторизацию


@jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
def _mandelbrot_kernel(c_real: np.ndarray, c_imag: np.ndarray, max_iterations: int) -> np.ndarray:
    """
    Ядро вычисления множества Мандельброта с использованием Numba JIT.
    """
    height, width = c_real.shape
    result = np.zeros((height, width), dtype=np.int32)
    
    for i in range(height):
        for j in range(width):
            c_r = c_real[i, j]
            c_i = c_imag[i, j]
            z_r = 0.0
            z_i = 0.0
            n = 0
            
            while (z_r * z_r + z_i * z_i <= 4.0) and (n < max_iterations):
                temp_r = z_r * z_r - z_i * z_i + c_r
                z_i = 2.0 * z_r * z_i + c_i
                z_r = temp_r
                n += 1
            
            result[i, j] = n
    
    return result


@jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
def _julia_kernel(z_real: np.ndarray, z_imag: np.ndarray, c_r: float, c_i: float, max_iterations: int) -> np.ndarray:
    """
    Ядро вычисления множества Жюлиа с использованием Numba JIT.
    """
    height, width = z_real.shape
    result = np.zeros((height, width), dtype=np.int32)
    
    for i in range(height):
        for j in range(width):
            z_r = z_real[i, j]
            z_i = z_imag[i, j]
            n = 0
            
            while (z_r * z_r + z_i * z_i <= 4.0) and (n < max_iterations):
                temp_r = z_r * z_r - z_i * z_i + c_r
                z_i = 2.0 * z_r * z_i + c_i
                z_r = temp_r
                n += 1
            
            result[i, j] = n
    
    return result


def mandelbrot_set(
    width: int,
    height: int,
    center_x: float = -0.5,
    center_y: float = 0.0,
    zoom: float = 1.0,
    max_iterations: int = 100
) -> np.ndarray:
    """
    Генерация множества Мандельброта с использованием векторизованных операций NumPy.
    
    Множество Мандельброта — это множество комплексных чисел c, для которых функция
    f(z) = z² + c не расходится при итерации начиная с z = 0.
    
    Args:
        width: Ширина выходного изображения в пикселях.
        height: Высота выходного изображения в пикселях.
        center_x: X-координата центра вида.
        center_y: Y-координата центра вида.
        zoom: Уровень масштабирования (выше = больше приближение).
        max_iterations: Максимальное количество итераций для определения расходимости.
                       Более высокие значения дают больше деталей, но работают медленнее.
    
    Returns:
        Двумерный массив numpy, где каждое значение представляет количество итераций
        до расходимости (или max_iterations, если не разошлось).
    """
    # Вычисляем границы на основе центра и масштаба
    x_min = center_x - 2.0 / zoom
    x_max = center_x + 2.0 / zoom
    y_min = center_y - 2.0 / zoom
    y_max = center_y + 2.0 / zoom
    
    # Создаем сетки координат
    x = np.linspace(x_min, x_max, width, dtype=np.float64)
    y = np.linspace(y_min, y_max, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    
    # Инициализируем комплексную плоскость
    C = X + 1j * Y
    Z = np.zeros_like(C, dtype=np.complex128)
    
    # Массив для хранения количества итераций
    iterations = np.zeros(C.shape, dtype=np.int32)
    
    # Маска активных точек (которые еще не разошлись)
    mask = np.ones(C.shape, dtype=bool)
    
    # Векторизованная итерация
    for i in range(max_iterations):
        Z[mask] = Z[mask] ** 2 + C[mask]
        diverged = np.abs(Z) > 2.0
        
        # Обновляем только точки, которые разошлись в этой итерации
        newly_diverged = diverged & mask
        iterations[newly_diverged] = i
        
        # Обновляем маску
        mask = mask & ~diverged
        
        # Если все точки разошлись, прекращаем
        if not np.any(mask):
            break
    
    # Точки, которые не разошлись, получают максимальное значение
    iterations[mask] = max_iterations
    
    return iterations


def julia_set(
    width: int,
    height: int,
    c_real: float = -0.7,
    c_imag: float = 0.27015,
    center_x: float = 0.0,
    center_y: float = 0.0,
    zoom: float = 1.0,
    max_iterations: int = 100
) -> np.ndarray:
    """
    Генерация множества Жюлиа с использованием векторизованных операций NumPy.
    
    Множество Жюлиа определяется итерацией функции f(z) = z² + c для фиксированного c.
    
    Args:
        width: Ширина выходного изображения в пикселях.
        height: Высота выходного изображения в пикселях.
        c_real: Действительная часть константы c.
        c_imag: Мнимая часть константы c.
        center_x: X-координата центра вида.
        center_y: Y-координата центра вида.
        zoom: Уровень масштабирования.
        max_iterations: Максимальное количество итераций.
    
    Returns:
        Двумерный массив numpy с количеством итераций до расходимости.
    """
    # Вычисляем границы
    x_min = center_x - 2.0 / zoom
    x_max = center_x + 2.0 / zoom
    y_min = center_y - 2.0 / zoom
    y_max = center_y + 2.0 / zoom
    
    # Создаем сетки координат
    x = np.linspace(x_min, x_max, width, dtype=np.float64)
    y = np.linspace(y_min, y_max, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    
    # Инициализируем Z как начальные точки
    Z = X + 1j * Y
    C = complex(c_real, c_imag)
    
    # Массив для хранения количества итераций
    iterations = np.zeros(Z.shape, dtype=np.int32)
    
    # Маска активных точек
    mask = np.ones(Z.shape, dtype=bool)
    
    # Векторизованная итерация
    for i in range(max_iterations):
        Z[mask] = Z[mask] ** 2 + C
        diverged = np.abs(Z) > 2.0
        
        newly_diverged = diverged & mask
        iterations[newly_diverged] = i
        
        mask = mask & ~diverged
        
        if not np.any(mask):
            break
    
    iterations[mask] = max_iterations
    
    return iterations


def fractal_tree(
    width: int,
    height: int,
    branch_angle: float = np.pi / 6,
    branch_ratio: float = 0.7,
    depth: int = 9,
    trunk_length: float = None
) -> np.ndarray:
    """
    Генерация фрактального дерева с использованием рекурсии.
    
    Фрактальное дерево создается путем рекурсивного ветвления линий.
    
    Args:
        width: Ширина выходного изображения в пикселях.
        height: Высота выходного изображения в пикселях.
        branch_angle: Угол ветвления в радианах.
        branch_ratio: Коэффициент уменьшения длины ветвей (0-1).
        depth: Глубина рекурсии (количество уровней ветвления).
        trunk_length: Длина ствола (по умолчанию высота/4).
    
    Returns:
        Двумерный массив numpy, где значения представляют глубину/интенсивность ветви.
    """
    if trunk_length is None:
        trunk_length = height / 4.0
    
    # Создаем пустое изображение
    tree_data = np.zeros((height, width), dtype=np.float32)
    
    # Центр внизу изображения
    start_x = width / 2
    start_y = height - 50
    
    def draw_branch(x, y, length, angle, current_depth):
        """Рекурсивная функция для рисования ветвей."""
        if current_depth == 0 or length < 2:
            return
        
        # Вычисляем конечную точку ветви
        end_x = x + length * np.sin(angle)
        end_y = y - length * np.cos(angle)
        
        # Рисуем ветвь (используем простую растеризацию линии)
        _draw_line(tree_data, x, y, end_x, end_y, current_depth, depth)
        
        # Рекурсивно рисуем дочерние ветви
        new_length = length * branch_ratio
        draw_branch(end_x, end_y, new_length, angle - branch_angle, current_depth - 1)
        draw_branch(end_x, end_y, new_length, angle + branch_angle, current_depth - 1)
    
    # Начинаем с ствола
    draw_branch(start_x, start_y, trunk_length, 0, depth)
    
    return tree_data


def _draw_line(data: np.ndarray, x0: float, y0: float, x1: float, y1: float, depth: int, max_depth: int):
    """
    Рисует линию в массиве данных, используя алгоритм Брезенхема.
    Значения зависят от глубины для создания градиента.
    """
    height, width = data.shape
    intensity = depth / max_depth
    
    x0, y0 = int(x0), int(y0)
    x1, y1 = int(x1), int(y1)
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    err = dx // 2 if dx > dy else -dy // 2
    
    x, y = x0, y0
    while True:
        if 0 <= y < height and 0 <= x < width:
            # Добавляем интенсивность (для перекрытия ветвей)
            data[y, x] = max(data[y, x], intensity)
        
        if x == x1 and y == y1:
            break
        
        e2 = err
        if e2 > -dx:
            err -= dy
            x += sx
        if e2 < dy:
            err += dx
            y += sy


# Кэш для результатов рендеринга
_render_cache = {}


def get_cached_fractal(cache_key: str) -> Optional[np.ndarray]:
    """Получить фрактал из кэша по ключу."""
    return _render_cache.get(cache_key)


def cache_fractal(cache_key: str, data: np.ndarray):
    """Сохранить фрактал в кэш."""
    _render_cache[cache_key] = data


def clear_cache():
    """Очистить весь кэш."""
    _render_cache.clear()


def get_fractal_parameters(fractal_type: str) -> Dict[str, Any]:
    """
    Получить параметры для указанного типа фрактала.
    
    Args:
        fractal_type: Тип фрактала ("mandelbrot", "julia", "tree").
    
    Returns:
        Словарь с параметрами, их описаниями, диапазонами и значениями по умолчанию.
    """
    params = {
        "mandelbrot": {
            "zoom": {
                "label": "Масштаб",
                "description": "Чем выше значение, тем ближе приближение к фракталу",
                "min": 0.1,
                "max": 1000.0,
                "default": 1.0,
                "step": 0.1,
                "type": "float"
            },
            "center_x": {
                "label": "Центр X",
                "description": "Горизонтальная позиция центра вида",
                "min": -2.5,
                "max": 1.0,
                "default": -0.5,
                "step": 0.01,
                "type": "float"
            },
            "center_y": {
                "label": "Центр Y",
                "description": "Вертикальная позиция центра вида",
                "min": -1.5,
                "max": 1.5,
                "default": 0.0,
                "step": 0.01,
                "type": "float"
            },
            "max_iterations": {
                "label": "Итерации",
                "description": "Больше = больше деталей, но медленнее. Рекомендуется 50-500",
                "min": 10,
                "max": 1000,
                "default": 100,
                "step": 10,
                "type": "int"
            }
        },
        "julia": {
            "c_real": {
                "label": "Действительная часть (c)",
                "description": "Вещественная компонента константы Жюлиа. Меняет форму фрактала",
                "min": -2.0,
                "max": 2.0,
                "default": -0.7,
                "step": 0.01,
                "type": "float"
            },
            "c_imag": {
                "label": "Мнимая часть (c)",
                "description": "Мнимая компонента константы Жюлиа. Меняет форму фрактала",
                "min": -2.0,
                "max": 2.0,
                "default": 0.27015,
                "step": 0.01,
                "type": "float"
            },
            "zoom": {
                "label": "Масштаб",
                "description": "Чем выше значение, тем ближе приближение",
                "min": 0.1,
                "max": 100.0,
                "default": 1.0,
                "step": 0.1,
                "type": "float"
            },
            "center_x": {
                "label": "Центр X",
                "description": "Горизонтальная позиция центра вида",
                "min": -2.0,
                "max": 2.0,
                "default": 0.0,
                "step": 0.01,
                "type": "float"
            },
            "center_y": {
                "label": "Центр Y",
                "description": "Вертикальная позиция центра вида",
                "min": -2.0,
                "max": 2.0,
                "default": 0.0,
                "step": 0.01,
                "type": "float"
            },
            "max_iterations": {
                "label": "Итерации",
                "description": "Больше = больше деталей, но медленнее",
                "min": 10,
                "max": 500,
                "default": 100,
                "step": 10,
                "type": "int"
            }
        },
        "tree": {
            "branch_angle": {
                "label": "Угол ветвления",
                "description": "Угол между ветвями в радианах. π/6 = 30°, π/4 = 45°",
                "min": 0.1,
                "max": np.pi / 2,
                "default": np.pi / 6,
                "step": 0.05,
                "type": "float"
            },
            "branch_ratio": {
                "label": "Коэффициент ветвей",
                "description": "Во сколько раз уменьшается каждая ветвь (0.5-0.9). Меньше = компактнее",
                "min": 0.5,
                "max": 0.9,
                "default": 0.7,
                "step": 0.01,
                "type": "float"
            },
            "depth": {
                "label": "Глубина рекурсии",
                "description": "Количество уровней ветвления. Больше = детальнее, но медленнее",
                "min": 5,
                "max": 12,
                "default": 9,
                "step": 1,
                "type": "int"
            },
            "trunk_length": {
                "label": "Длина ствола",
                "description": "Начальная длина главного ствола в пикселях",
                "min": 50,
                "max": 300,
                "default": 150,
                "step": 10,
                "type": "int"
            }
        }
    }
    
    return params.get(fractal_type, {})


def generate_fractal(
    fractal_type: str,
    width: int,
    height: int,
    params: Dict[str, Any],
    use_cache: bool = True,
    cache_key: str = None
) -> np.ndarray:
    """
    Основная функция для генерации фракталов.
    
    Args:
        fractal_type: Тип фрактала ("mandelbrot", "julia", "tree").
        width: Ширина изображения.
        height: Высота изображения.
        params: Параметры для фрактала.
        use_cache: Использовать ли кэширование.
        cache_key: Ключ для кэширования (если None, генерируется автоматически).
    
    Returns:
        Двумерный массив numpy с данными фрактала.
    """
    # Генерируем ключ кэша если не предоставлен
    if cache_key is None:
        cache_key = f"{fractal_type}_{width}x{height}_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    
    # Проверяем кэш
    if use_cache:
        cached = get_cached_fractal(cache_key)
        if cached is not None:
            return cached
    
    # Генерируем фрактал
    if fractal_type == "mandelbrot":
        data = mandelbrot_set(
            width=width,
            height=height,
            center_x=params.get("center_x", -0.5),
            center_y=params.get("center_y", 0.0),
            zoom=params.get("zoom", 1.0),
            max_iterations=params.get("max_iterations", 100)
        )
    elif fractal_type == "julia":
        data = julia_set(
            width=width,
            height=height,
            c_real=params.get("c_real", -0.7),
            c_imag=params.get("c_imag", 0.27015),
            center_x=params.get("center_x", 0.0),
            center_y=params.get("center_y", 0.0),
            zoom=params.get("zoom", 1.0),
            max_iterations=params.get("max_iterations", 100)
        )
    elif fractal_type == "tree":
        data = fractal_tree(
            width=width,
            height=height,
            branch_angle=params.get("branch_angle", np.pi / 6),
            branch_ratio=params.get("branch_ratio", 0.7),
            depth=params.get("depth", 9),
            trunk_length=params.get("trunk_length", 150)
        )
    else:
        raise ValueError(f"Неизвестный тип фрактала: {fractal_type}")
    
    # Сохраняем в кэш
    if use_cache:
        cache_fractal(cache_key, data)
    
    return data
