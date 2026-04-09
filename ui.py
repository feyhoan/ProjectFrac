"""
ui.py - User interface using tkinter with a modern, clean layout.

This module contains:
- Main Menu with fractal preset cards
- Workspace window with canvas and control panel
- Threading for non-blocking rendering
"""

import tkinter as tk
from tkinter import ttk, colorchooser
from PIL import Image, ImageTk
import threading
from typing import Optional, Dict, Any, List, Tuple

# Import our modules
from renderer import create_placeholder_image, get_available_palettes, render_fractal
from core import get_fractal_parameters


class FractalApp:
    """Main application class for the Fractal Visualizer."""
    
    def __init__(self):
        """Initialize the application."""
        self.root = tk.Tk()
        self.root.title("Fractal Visualizer")
        self.root.geometry("1200x800")
        
        # Set modern style
        self._setup_styles()
        
        # Current state
        self.current_fractal_type: Optional[str] = None
        self.current_params: Dict[str, Any] = {}
        self.current_palette: str = "Classic"
        self.background_color: Optional[Tuple[int, int, int]] = None
        
        # Control widgets storage
        self.param_widgets: Dict[str, Any] = {}
        
        # Show main menu
        self.show_main_menu()
    
    def _setup_styles(self):
        """Configure modern styling for tkinter widgets."""
        # Configure colors
        self.colors = {
            "bg_dark": "#1a1a2e",
            "bg_medium": "#16213e",
            "bg_light": "#0f3460",
            "accent": "#e94560",
            "text_primary": "#ffffff",
            "text_secondary": "#b0b0b0",
            "card_bg": "#252540",
            "button_bg": "#e94560",
            "button_fg": "#ffffff"
        }
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure frame background
        style.configure(
            "TFrame",
            background=self.colors["bg_dark"]
        )
        
        # Configure labels
        style.configure(
            "TLabel",
            background=self.colors["bg_dark"],
            foreground=self.colors["text_primary"],
            font=("Segoe UI", 10)
        )
        
        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 14, "bold"),
            foreground=self.colors["accent"]
        )
        
        style.configure(
            "Description.TLabel",
            font=("Segoe UI", 9),
            foreground=self.colors["text_secondary"]
        )
        
        # Configure buttons
        style.configure(
            "TButton",
            background=self.colors["button_bg"],
            foreground=self.colors["button_fg"],
            font=("Segoe UI", 10, "bold"),
            padding=10
        )
        
        style.map(
            "TButton",
            background=[("active", "#ff6b7a"), ("pressed", "#c73e54")]
        )
        
        # Configure combobox
        style.configure(
            "TCombobox",
            fieldbackground=self.colors["bg_medium"],
            background=self.colors["bg_light"],
            foreground=self.colors["text_primary"],
            arrowcolor=self.colors["text_primary"]
        )
        
        # Configure scales (sliders) - these are harder to style in tkinter
        # We'll use custom colors when creating them
    
    def show_main_menu(self):
        """Display the main menu with fractal selection cards."""
        # Clear any existing content
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="🎨 Fractal Visualizer",
            style="Title.TLabel"
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(
            main_frame,
            text="Select a fractal type to begin exploring",
            style="Description.TLabel"
        )
        subtitle_label.pack(pady=(0, 40))
        
        # Cards container
        cards_frame = ttk.Frame(main_frame)
        cards_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create three fractal cards
        fractals = [
            ("Mandelbrot Set", "mandelbrot", "#e94560"),
            ("Julia Set", "julia", "#0f3460"),
            ("Fractal Tree", "tree", "#533483")
        ]
        
        for name, fractal_type, card_color in fractals:
            card = self._create_fractal_card(cards_frame, name, fractal_type, card_color)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20)
    
    def _create_fractal_card(
        self, 
        parent, 
        name: str, 
        fractal_type: str, 
        card_color: str
    ) -> ttk.Frame:
        """Create a clickable card for fractal selection."""
        card = ttk.Frame(parent, style="TFrame")
        card.pack_propagate(False)
        card.config(width=280, height=350)
        
        # Inner card with border effect
        inner_frame = tk.Frame(
            card,
            bg=card_color,
            highlightthickness=2,
            highlightbackground=self.colors["bg_light"]
        )
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate placeholder image
        placeholder = create_placeholder_image(
            text=name,
            width=260,
            height=180,
            bg_color=tuple(int(card_color[i:i+2], 16) if i < len(card_color) else 100 
                          for i in (1, 3, 5)),
            text_color=(255, 255, 255)
        )
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(placeholder)
        
        # Image label
        image_label = tk.Label(inner_frame, image=photo, bg=card_color)
        image_label.image = photo  # Keep reference to prevent garbage collection
        image_label.pack(pady=20)
        
        # Name label
        name_label = tk.Label(
            inner_frame,
            text=name,
            font=("Segoe UI", 16, "bold"),
            bg=card_color,
            fg=self.colors["text_primary"]
        )
        name_label.pack(pady=10)
        
        # Description
        descriptions = {
            "Mandelbrot Set": "The most famous fractal.\nInfinite complexity at every scale.",
            "Julia Set": "Related to Mandelbrot.\nVary the constant c for different shapes.",
            "Fractal Tree": "Recursive branching pattern.\nNature-inspired mathematical beauty."
        }
        
        desc_label = tk.Label(
            inner_frame,
            text=descriptions.get(name, ""),
            font=("Segoe UI", 10),
            bg=card_color,
            fg=self.colors["text_secondary"],
            justify=tk.CENTER
        )
        desc_label.pack(pady=10)
        
        # Make card clickable
        card.bind("<Button-1>", lambda e: self.open_workspace(fractal_type))
        inner_frame.bind("<Button-1>", lambda e: self.open_workspace(fractal_type))
        name_label.bind("<Button-1>", lambda e: self.open_workspace(fractal_type))
        image_label.bind("<Button-1>", lambda e: self.open_workspace(fractal_type))
        
        return card
    
    def open_workspace(self, fractal_type: str):
        """Open the workspace window for the selected fractal."""
        self.current_fractal_type = fractal_type
        
        # Clear main menu
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Create workspace
        self._create_workspace()
        
        # Initialize default parameters
        params_config = get_fractal_parameters(fractal_type)
        for param_name, config in params_config.items():
            self.current_params[param_name] = config["default"]
    
    def _create_workspace(self):
        """Create the workspace layout with canvas and control panel."""
        # Main horizontal paned window (70% canvas, 30% controls)
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Canvas area
        canvas_frame = ttk.Frame(paned)
        paned.add(canvas_frame, weight=70)
        
        # Canvas for displaying the fractal
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#000000",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right side: Control panel
        control_frame = ttk.Frame(paned)
        paned.add(control_frame, weight=30)
        
        # Create control panel content
        self._create_control_panel(control_frame)
    
    def _create_control_panel(self, parent):
        """Create the control panel with parameters and settings."""
        # Scrollable frame for controls
        canvas = tk.Canvas(parent, bg=self.colors["bg_dark"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        
        self.controls_frame = ttk.Frame(canvas)
        
        canvas_window = canvas.create_window((0, 0), window=self.controls_frame, anchor=tk.NW)
        
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        self.controls_frame.bind("<Configure>", configure_scroll_region)
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Title
        title = ttk.Label(
            self.controls_frame,
            text=f"{self.current_fractal_type.title()} Controls",
            style="Title.TLabel"
        )
        title.pack(pady=(20, 10), padx=15)
        
        # Parameters section
        params_frame = ttk.LabelFrame(
            self.controls_frame,
            text="Parameters",
            padding=10
        )
        params_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Create parameter controls
        self._create_parameter_controls(params_frame)
        
        # Color settings section
        color_frame = ttk.LabelFrame(
            self.controls_frame,
            text="Color Settings",
            padding=10
        )
        color_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self._create_color_controls(color_frame)
        
        # Render button section
        button_frame = ttk.Frame(self.controls_frame)
        button_frame.pack(fill=tk.X, padx=15, pady=20)
        
        self.render_button = tk.Button(
            button_frame,
            text="RENDER",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors["button_bg"],
            fg=self.colors["button_fg"],
            activebackground="#ff6b7a",
            activeforeground=self.colors["button_fg"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_rendering
        )
        self.render_button.pack(fill=tk.X, ipady=15)
        
        # Back button
        back_button = tk.Button(
            button_frame,
            text="← Back to Menu",
            font=("Segoe UI", 10),
            bg=self.colors["bg_light"],
            fg=self.colors["text_primary"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self.show_main_menu
        )
        back_button.pack(fill=tk.X, ipady=8, pady=10)
    
    def _create_parameter_controls(self, parent):
        """Create controls for fractal parameters."""
        params_config = get_fractal_parameters(self.current_fractal_type)
        
        for param_name, config in params_config.items():
            # Container for each parameter
            param_frame = ttk.Frame(parent)
            param_frame.pack(fill=tk.X, pady=8)
            
            # Label with description
            label_text = param_name.replace("_", " ").title()
            label = ttk.Label(
                param_frame,
                text=label_text,
                style="TLabel"
            )
            label.pack(anchor=tk.W)
            
            # Description label
            desc_label = ttk.Label(
                param_frame,
                text=config["description"],
                style="Description.TLabel",
                wraplength=250
            )
            desc_label.pack(anchor=tk.W)
            
            # Min/Max info for scales
            if config["type"] == "scale":
                range_label = ttk.Label(
                    param_frame,
                    text=f"Range: {config['min']} - {config['max']}",
                    style="Description.TLabel"
                )
                range_label.pack(anchor=tk.W)
            
            # Value display
            value_var = tk.StringVar(value=str(config["default"]))
            
            if config["type"] == "scale":
                # Slider control
                slider_frame = ttk.Frame(param_frame)
                slider_frame.pack(fill=tk.X, pady=5)
                
                slider = tk.Scale(
                    slider_frame,
                    from_=config["min"],
                    to=config["max"],
                    resolution=config["step"],
                    orient=tk.HORIZONTAL,
                    length=200,
                    bg=self.colors["bg_medium"],
                    fg=self.colors["text_primary"],
                    troughcolor=self.colors["bg_light"],
                    highlightthickness=0,
                    command=lambda v, name=param_name: self._update_param(name, v)
                )
                slider.set(config["default"])
                slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
                
                # Value entry for precise input
                value_entry = tk.Entry(
                    slider_frame,
                    textvariable=value_var,
                    width=8,
                    bg=self.colors["bg_medium"],
                    fg=self.colors["text_primary"],
                    insertbackground=self.colors["text_primary"],
                    relief=tk.FLAT
                )
                value_entry.pack(side=tk.LEFT, padx=(10, 0))
                value_entry.bind(
                    "<Return>",
                    lambda e, name=param_name: self._update_param_from_entry(name)
                )
                
                self.param_widgets[param_name] = {
                    "slider": slider,
                    "entry": value_entry,
                    "value_var": value_var
                }
            
            else:  # Entry type
                entry = tk.Entry(
                    param_frame,
                    textvariable=value_var,
                    bg=self.colors["bg_medium"],
                    fg=self.colors["text_primary"],
                    insertbackground=self.colors["text_primary"],
                    relief=tk.FLAT
                )
                entry.pack(fill=tk.X, pady=5)
                entry.bind(
                    "<Return>",
                    lambda e, name=param_name: self._update_param_from_entry(name)
                )
                
                self.param_widgets[param_name] = {
                    "entry": entry,
                    "value_var": value_var
                }
    
    def _create_color_controls(self, parent):
        """Create color selection controls."""
        # Palette selector
        palette_frame = ttk.Frame(parent)
        palette_frame.pack(fill=tk.X, pady=8)
        
        ttk.Label(
            palette_frame,
            text="Color Palette:",
            style="TLabel"
        ).pack(anchor=tk.W)
        
        palettes = get_available_palettes()
        self.palette_var = tk.StringVar(value="Classic")
        
        palette_combo = ttk.Combobox(
            palette_frame,
            textvariable=self.palette_var,
            values=palettes,
            state="readonly"
        )
        palette_combo.pack(fill=tk.X, pady=5)
        palette_combo.bind("<<ComboboxSelected>>", self._on_palette_change)
        
        # Background color picker
        bg_frame = ttk.Frame(parent)
        bg_frame.pack(fill=tk.X, pady=8)
        
        ttk.Label(
            bg_frame,
            text="Background Color:",
            style="TLabel"
        ).pack(anchor=tk.W)
        
        bg_button = tk.Button(
            bg_frame,
            text="Choose Background Color",
            font=("Segoe UI", 9),
            bg=self.colors["bg_medium"],
            fg=self.colors["text_primary"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._choose_background_color
        )
        bg_button.pack(fill=tk.X, pady=5)
        
        # Reset background button
        reset_bg_button = tk.Button(
            bg_frame,
            text="Reset to Default",
            font=("Segoe UI", 8),
            bg=self.colors["bg_light"],
            fg=self.colors["text_secondary"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._reset_background_color
        )
        reset_bg_button.pack(fill=tk.X)
    
    def _update_param(self, param_name: str, value: str):
        """Update parameter value from slider."""
        try:
            self.current_params[param_name] = float(value)
            # Update entry widget
            if param_name in self.param_widgets:
                self.param_widgets[param_name]["value_var"].set(str(value))
        except ValueError:
            pass
    
    def _update_param_from_entry(self, param_name: str):
        """Update parameter value from entry widget."""
        try:
            value = self.param_widgets[param_name]["value_var"].get()
            self.current_params[param_name] = float(value)
            # Update slider if exists
            if "slider" in self.param_widgets[param_name]:
                self.param_widgets[param_name]["slider"].set(float(value))
        except ValueError:
            pass
    
    def _on_palette_change(self, event=None):
        """Handle palette selection change."""
        self.current_palette = self.palette_var.get()
    
    def _choose_background_color(self):
        """Open color chooser for background color."""
        color = colorchooser.askcolor(
            title="Choose Background Color",
            parent=self.root
        )
        if color[0]:  # RGB tuple
            self.background_color = tuple(int(c) for c in color[0])
    
    def _reset_background_color(self):
        """Reset background color to default (None)."""
        self.background_color = None
    
    def start_rendering(self):
        """Start the rendering process in a background thread."""
        # Disable button and show loading animation
        self.render_button.config(state=tk.DISABLED, text="Rendering... |")
        self.render_animation_frame = 0
        self.render_animation_chars = ["|", "/", "-", "\\"]
        
        def animate_button():
            if self.render_button.cget("state") == tk.DISABLED:
                char = self.render_animation_chars[self.render_animation_frame % 4]
                self.render_button.config(text=f"Rendering... {char}")
                self.render_animation_frame += 1
                self.root.after(150, animate_button)
        
        animate_button()
        
        # Start rendering in background thread
        thread = threading.Thread(target=self._render_thread, daemon=True)
        thread.start()
    
    def _render_thread(self):
        """Background thread for rendering the fractal."""
        try:
            # Get canvas size
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            
            # Ensure minimum size
            if width < 100 or height < 100:
                width = 800
                height = 600
            
            # Render the fractal
            image = render_fractal(
                fractal_type=self.current_fractal_type,
                params=self.current_params,
                width=width,
                height=height,
                palette_name=self.current_palette,
                background_color=self.background_color
            )
            
            # Convert to PhotoImage and display on main thread
            photo = ImageTk.PhotoImage(image)
            
            # Schedule update on main thread
            self.root.after(0, lambda: self._display_image(photo))
            
        except Exception as e:
            # Handle errors on main thread
            self.root.after(0, lambda: self._render_error(str(e)))
    
    def _display_image(self, photo: ImageTk.PhotoImage):
        """Display the rendered image on the canvas."""
        # Store reference to prevent garbage collection
        self.current_image = photo
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        # Reset button
        self.render_button.config(state=tk.NORMAL, text="RENDER")
    
    def _render_error(self, error_msg: str):
        """Handle rendering errors."""
        print(f"Rendering error: {error_msg}")
        
        # Display error on canvas
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() / 2,
            self.canvas.winfo_height() / 2,
            text=f"Error:\n{error_msg}",
            fill="red",
            font=("Segoe UI", 14),
            justify=tk.CENTER
        )
        
        # Reset button
        self.render_button.config(state=tk.NORMAL, text="RENDER")
    
    def run(self):
        """Start the application main loop."""
        self.root.mainloop()


def main():
    """Entry point for the application."""
    app = FractalApp()
    app.run()


if __name__ == "__main__":
    main()
