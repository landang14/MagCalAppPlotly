from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shinywidgets import output_widget, render_plotly
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
from pathlib import Path
import math
import plotly.graph_objects as go
from plotly.graph_objects import FigureWidget
from compute import (
    fit_ellipse_fixed_center,
    normalize,
    normalize_image,
    read_mrc_as_image,
    load_image,
    fft_image_with_matplotlib,
    compute_fft_image_region,
    compute_average_fft,
    calculate_apix_from_distance,
    calculate_distance_from_apix,
    calculate_tilt_angle,
    get_resolution_info,
    compute_fft_1d_data,
    resolution_to_radius,
    create_fft_1d_plotly_figure,
    get_image,
    plot_image,
    get_image_with_binning,
    extract_region_from_original,
    extract_region_no_binning,
    bin_image
)
# ---------- Documentation ----------
"""Magnification Calibration Tool

This tool helps calibrate electron microscopes by analyzing test specimen images.
It calculates the pixel size (Angstroms/pixel) by measuring diffraction patterns
from known specimens like graphene, gold, or ice.

Key Features:
- Supports common image formats (.png, .tif) and MRC files
- Interactive FFT analysis with resolution circles
- Automatic pixel size detection
- Radial averaging for enhanced signal detection
- Customizable resolution measurements

Usage:
1. Upload a test specimen image
2. Select the expected diffraction pattern (graphene/gold/ice)
3. Adjust the region size to analyze
4. Click points in the FFT to measure distances
5. Use auto-search to find the best pixel size match

The tool will display:
- Original image with selected region
- FFT with resolution circles
- 1D radial average plot
- Calculated pixel size (Angstroms/pixel)
"""
import argparse

def print_help():
    """Print usage instructions and help information."""
    help_text = """
Magnification Calibration Tool
---------------------------

Usage:
    Run the Shiny app and follow the web interface.
    
Input Files:
    - Image formats: PNG, TIFF
    - MRC files from microscopes
    
Key Parameters:
    Apix: Pixel size in Angstroms/pixel (0.01-6.0)
    Region: Size of FFT analysis region (1-100%)
    Resolution circles:
        - Graphene: 2.13 Å
        - Gold: 2.355 Å 
        - Ice: 3.661 Å
        - Custom: User-defined resolution
        
Analysis Features:
    - Interactive FFT region selection
    - Resolution circle overlay
    - Automatic pixel size detection
    - Radial averaging
    - Click-to-measure distances
    
Output:
    - Processed FFT image
    - Radial intensity profile
    - Calculated pixel size
    """
    print(help_text)
    
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("upload", "Upload an image of test specimens (e.g., graphene)(.mrc,.tiff,.png)", accept=["image/*", ".mrc", ".tif", ".png"]),
        ui.input_select("resolution_type", "Resolution Type", 
                      choices=["Graphene (2.13 Å)", "Gold (2.355 Å)", "Ice (3.661 Å)", "Custom"], 
                      selected="Graphene (2.13 Å)"),
        ui.panel_conditional(
            "input.resolution_type == 'Custom'",
            ui.div(
                {"style": "display: flex; align-items: center;"},
                ui.input_numeric("custom_resolution", "Custom Res (Å):", value=3.0, min=0.1, max=10.0, step=0.01, width="80px"),
            ),
        ),
        ui.input_select("label_mode", "Label", 
                      choices=["Resolution Ring", "Lattice Point"], 
                      selected="Resolution Ring"),
        ui.div(
            {"style": "padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 10px; display: flex; flex-direction: column; gap: 5px;"},
            ui.div(
                {"style": "flex: 1;"},
                ui.input_slider("apix_slider", "Apix (Å/px)", min=0.01, max=2.0, value=1.0, step=0.001),
            ),
            ui.div(
                {"style": "display: flex; justify-content: flex-start; align-items: bottom; gap: 5px; margin-top: 5px; width: 100%;"},
                ui.input_text("apix_exact_str", None, value="1.0", width="70px"),
                ui.input_action_button("apix_set_btn", ui.tags.span("Set", style="display: flex; align-items: center; justify-content: center; width: 100%; height: 100%;"), class_="btn-primary", style="height: 38px; display: flex; align-items: center;", width="50px"),
            ),
        ),

        ui.div(
            {"style": "display: none;"},  # Hidden div for data persistence
            ui.output_text("lattice_points_data"),
        ),
        
        # FFT Analysis Controls
        ui.h3("FFT Analysis Controls", style="margin-top: 20px; margin-bottom: 10px;"),
        ui.input_slider("zoom2", "FFT Zoom (%)", min=50, max=300, value=100),
        ui.input_slider("contrast", "FFT Range (±σ)", min=0.1, max=5.0, value=2.0, step=0.1),
        ui.output_text("tilt_output"),
        
        title=ui.h2("Magnification Calibration", style="font-size: 36px; font-weight: bold; padding: 15px;"),
        open="open",
        width="400px",
        min_width="250px",
        max_width="500px",
        resize=True,
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Original Image"),
            output_widget("image_display"),
            ui.div(
                {"style": "display: flex; gap: 10px; padding: 10px; justify-content: center;"},
                ui.input_action_button("clear_drawn_region", "Clear Region", class_="btn-secondary"),
                ui.input_action_button("calc_fft", "Calc FFT", class_="btn-primary"),
            ),
            ui.div(
                {"class": "card-footer"},
                "Drag to zoom in and select regions for FFT analysis.",
            ),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("FFT Analysis"),
            output_widget("fft_with_circle"),
            ui.div(
                {"style": "display: flex; gap: 10px; padding: 10px; justify-content: center;"},
                ui.input_action_button("clear_markers", "Clear Markers", class_="btn-secondary"),
                ui.input_action_button("fit_markers", "Fit Ellipse", class_="btn-secondary"),
                ui.input_action_button("estimate_tilt", "Estimate Tilt", class_="btn-secondary"),
            ),
            ui.div(
                {"class": "card-footer"},
                "Click to mark points based on selected Label mode in sidebar.",
            ),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("1D FFT Radial Profile"),

            ui.div(
                {"style": "display: flex;"},
                output_widget("fft_1d_plot"),
                ui.div(
                    {"style": "display: flex; flex-direction: column; justify-content: flex-start; margin-left: 10px; width: 200px;"},
                    ui.input_checkbox("log_y", "Log Scale", value=False),
                    ui.input_checkbox("use_mean_profile", "Use Average Profile", value=False),
                    ui.input_checkbox("smooth", "Smooth Signal", value=False),
                    ui.input_checkbox("detrend", "Detrend Signal", value=False),
                    ui.div(
                        {"style": "margin-bottom: 10px;"},
                        ui.panel_conditional(
                            "input.smooth",
                            ui.input_slider("window_size", "Window Size", min=1, max=11, value=3, step=2),
                        ),
                    ),
                    ui.input_action_button("reset_zoom", "Reset Zoom"),
                    ui.input_action_button("estimate_tilt_1d", "Estimate Tilt", class_="btn-secondary"),
                    ui.output_text("tilt_1d_output"),
                ),
            ),
            ui.div(
                {"class": "card-footer", "style": "justify-content: flex-start;"},
                "Radial Max of the 2D FFT. Drag to zoom, double-click to reset."
            ),
            full_screen=True,
        ),
        col_widths=[6, 6, 12],
    ),

    fillable=True,
)
# Add custom CSS for layout
app_ui = ui.tags.div(
    ui.tags.style("""
        /* Image container styles */
        .image-output {
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
            padding: 10px;
            margin-bottom: 10px;
            width: 100%;
            min-height: 300px;
            flex: 1;
            /* Ensure scrollbars are always visible and not hidden */
            scrollbar-width: auto;
            scrollbar-color: rgba(0, 0, 0, 0.3) transparent;
        }
        .image-output::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }
        .image-output::-webkit-scrollbar-track {
            background: transparent;
        }
        .image-output::-webkit-scrollbar-thumb {
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 6px;
            border: 2px solid transparent;
        }
        .image-output img {
            height: auto;
            width: auto;
            max-width: none;
            max-height: none;
            /* Add margin to ensure scrollbar is not covered */
            margin-bottom: 12px;
        }
        /* Footer styles */
        .card-footer {
            height: 40px;
            padding: 8px;
            background-color: rgba(0, 0, 0, 0.03);
            border-top: 1px solid rgba(0, 0, 0, 0.125);
            display: flex;
            align-items: center;
            flex-shrink: 0;
            margin-top: 0;
            width: 100%;
        }
        .sidebar > h2, 
        .sidebar-title,
        .shiny-sidebar-title {
            font-size: 36px !important;
            font-weight: bold !important;
            padding: 15px !important;
        }
        
        /* Make Plotly widgets fill their containers */
        .js-plotly-plot {
            height: 100% !important;
            width: 100% !important;
        }
        
        /* Ensure cards with Plotly widgets use full height */
        .card .output_widget {
            height: 100%;
            min-height: 400px;
        }
        

    """),
    ui.tags.script("""
        // Custom JavaScript to ensure only one shape at a time
        document.addEventListener('DOMContentLoaded', function() {
            // Function to clear all shapes except the latest one
            function clearPreviousShapes() {
                const plots = document.querySelectorAll('.js-plotly-plot');
                plots.forEach(plot => {
                    if (plot.layout && plot.layout.shapes && plot.layout.shapes.length > 1) {
                        // Keep only the last shape
                        const lastShape = plot.layout.shapes[plot.layout.shapes.length - 1];
                        plot.layout.shapes = [lastShape];
                        Plotly.relayout(plot, {shapes: [lastShape]});
                    }
                });
            }
            
            // Listen for shape drawing events using MutationObserver
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList') {
                        const plots = document.querySelectorAll('.js-plotly-plot');
                        plots.forEach(plot => {
                            if (plot.layout && plot.layout.shapes && plot.layout.shapes.length > 1) {
                                setTimeout(clearPreviousShapes, 50);
                            }
                        });
                    }
                });
            });
            
            // Start observing
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            
            // Also listen for click events on plots
            document.addEventListener('click', function(e) {
                if (e.target.closest('.js-plotly-plot')) {
                    setTimeout(clearPreviousShapes, 100);
                }
            });
        });
    """),
    app_ui
)
size = 360

# ---------- Helper Functions ----------

# ---------- Server ----------
def server(input: Inputs, output: Outputs, session: Session):
    # Central reactive state for FFT panel
    fft_state = reactive.Value({
        'mode': 'Resolution Ring',
        'resolution_radius': None,
        'resolution_click_x': None,
        'resolution_click_y': None,
        'lattice_points': [],
        'ellipse_params': None,
        'tilt_info': None,
        'zoom_factor': 1.0,
        'plot_1d_markers': [],  # List of (x, y) tuples for 1D plot markers in Lattice Point mode
        'tilt_info_1d': None  # (smaller_x, larger_x, tilt_angle, apix_value) for 1D tilt estimation
    })
    
    # Separate reactive values for FFT image rendering to avoid unnecessary re-renders
    fft_markers = reactive.Value({
        'mode': 'Resolution Ring',
        'resolution_click_x': None,
        'resolution_click_y': None,
        'lattice_points': [],
        'ellipse_params': None,
        'zoom_factor': 1.0
    })
    


    # --- Single source of truth for apix ---
    apix_master = reactive.Value(1.0)

    # Effect to update fft_markers when relevant parts of fft_state change
    @reactive.Effect
    @reactive.event(fft_state)
    def _():
        """Update fft_markers when relevant parts of fft_state change."""
        state = fft_state.get()
        fft_markers.set({
            'mode': state['mode'],
            'resolution_click_x': state['resolution_click_x'],
            'resolution_click_y': state['resolution_click_y'],
            'lattice_points': state['lattice_points'].copy(),
            'ellipse_params': state['ellipse_params'],
            'zoom_factor': state['zoom_factor']
        })
    
    # Remove fft_1d_data since we're no longer using static markers

    # Add reactive value to cache the base FFT image
    cached_fft_image = reactive.Value(None)

    # Add reactive value to cache the FFT image without resolution circles
    cached_fft_image_no_circles = reactive.Value(None)

    # Add plot zoom state
    plot_zoom = reactive.Value({
        'x_range': None,
        'y_range': None
    })

    # Add reactive values for raw data and region
    raw_image_data = reactive.Value({
        'img': None,
        'data': None
    })

    # Add reactive values for image display
    image_data = reactive.Value(None)
    image_apix = reactive.Value(1.0)
    image_filename = reactive.Value(None)
    
    # Add reactive values for original and binned image data
    original_image_data = reactive.Value(None)
    binned_image_data = reactive.Value(None)
    
    # Add reactive value for image zoom state
    image_zoom_state = reactive.Value({
        'x_range': None,
        'y_range': None,
        'is_zoomed': False,
        'drawn_region': None  # Store drawn rectangle coordinates
    })

    # Add reactive value to trigger FFT calculations
    fft_trigger = reactive.Value(0)
    
    # Add reactive value to store the 1D plot FigureWidget for in-place updates
    fft_1d_widget = reactive.Value(None)
    
    # Add reactive value to store all drawn shapes
    drawn_shapes = reactive.Value([])

    # Initialize Fit button state
    @reactive.Effect
    def _():
        """Initialize Fit button state."""
        is_disabled = input.label_mode() != "Lattice Point"
        ui.update_action_button("fit_markers", disabled=is_disabled, session=session)
        ui.update_action_button("estimate_tilt", disabled=is_disabled, session=session)
        ui.update_action_button("estimate_tilt_1d", disabled=is_disabled, session=session)

    # --- All events update apix_master ---
    @reactive.Effect
    @reactive.event(input.apix_slider)
    def _():
        apix_master.set(input.apix_slider())
        # Clear 1D plot clicked position when apix changes from slider
        #plot_1d_click_pos.set({'x': None, 'y': None})

    @reactive.Effect
    @reactive.event(input.apix_set_btn)
    def _():
        try:
            val = float(input.apix_exact_str())
            if 0.001 <= val <= 6.0:
                #apix_master.set(val)
                ui.update_slider("apix_slider", value=val, session=session)
                ui.update_text("apix_exact_str", value=str(round(val, 3)), session=session)
                # Clear 1D plot clicked position when apix changes from Set button
                #plot_1d_click_pos.set({'x': None, 'y': None})
        except Exception:
            pass







    @reactive.Effect
    @reactive.event(input["fft_with_circle_click"])
    def _():
        click_data = input["fft_with_circle_click"]()
        if click_data is not None:
            zoom_factor = input.zoom2() / 100
            x = click_data['x'] / zoom_factor
            y = click_data['y'] / zoom_factor
            
            current_state = fft_state.get()
            
            if input.label_mode() == "Resolution Ring":
                # Resolution Ring mode: set crosshair and calculate radius
                resolution, color = get_resolution_info(input.resolution_type(), input.custom_resolution())
                if resolution is None:
                    return
                
                # Calculate distance from center in pixels
                center_x = size / 2
                center_y = size / 2
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Update state
                new_state = current_state.copy()
                new_state['mode'] = 'Resolution Ring'
                new_state['resolution_radius'] = distance
                new_state['resolution_click_x'] = x
                new_state['resolution_click_y'] = y
                new_state['lattice_points'] = []  # Clear lattice points
                fft_state.set(new_state)
                
                # Update apix if distance is valid
                if distance > 0:
                    new_apix = (distance * resolution) / size
                    if 0.01 <= new_apix <= 6.0:
                        new_apix = round(new_apix, 3)
                        ui.update_slider("apix_slider", value=new_apix, session=session)
                        ui.update_text("apix_exact_str", value=str(new_apix), session=session)
            
            elif input.label_mode() == "Lattice Point":
                # Lattice Point mode: add point to list
                new_state = current_state.copy()
                new_state['mode'] = 'Lattice Point'
                new_state['lattice_points'].append((x, y))
                fft_state.set(new_state)

    @reactive.Effect
    @reactive.event(input.clear_markers)
    def _():
        """Clear all markers based on current mode."""
        current_state = fft_state.get()
        new_state = current_state.copy()
        
        if current_state['mode'] == 'Resolution Ring':
            # Clear resolution ring markers
            new_state['resolution_radius'] = None
            new_state['resolution_click_x'] = None
            new_state['resolution_click_y'] = None
        elif current_state['mode'] == 'Lattice Point':
            # Clear lattice points, ellipse, tilt info, and 1D plot markers
            new_state['lattice_points'] = []
            new_state['ellipse_params'] = None
            new_state['tilt_info'] = None
            new_state['plot_1d_markers'] = []
            new_state['tilt_info_1d'] = None
        
        fft_state.set(new_state)

    @reactive.Effect
    @reactive.event(input.clear_drawn_region)
    def _():
        """Clear drawn region only."""
        current_zoom_state = image_zoom_state.get()
        new_zoom_state = current_zoom_state.copy()
        new_zoom_state['drawn_region'] = None
        image_zoom_state.set(new_zoom_state)
        drawn_shapes.set([])
        print("Drawn shapes cleared")



    @reactive.Effect
    @reactive.event(input.calc_fft)
    def _():
        """Manually trigger FFT calculation."""
        print("=== MANUAL FFT CALCULATION TRIGGERED ===")
        
        # Get the last drawn shape from our stored shapes
        shapes = drawn_shapes.get()
        print(f"Available shapes: {len(shapes)}")
        print(f"Shapes content: {shapes}")
        
        if shapes and len(shapes) > 0:
            # Get the last shape (most recent)
            latest_shape = shapes[-1]
            print(f"Using latest shape: {latest_shape}")
            
            if latest_shape.get('type') == 'rect':
                # Extract rectangle coordinates
                x0 = latest_shape.get('x0')
                x1 = latest_shape.get('x1')
                y0 = latest_shape.get('y0')
                y1 = latest_shape.get('y1')
                
                print(f"Raw coordinates: x0={x0}, x1={x1}, y0={y0}, y1={y1}")
                
                if all(coord is not None for coord in [x0, x1, y0, y1]):
                    # Enforce square constraint by using the smaller dimension
                    width = abs(x1 - x0)
                    height = abs(y1 - y0)
                    square_size = min(width, height)
                    
                    # Recalculate coordinates to make it a perfect square
                    center_x = (x0 + x1) / 2
                    center_y = (y0 + y1) / 2
                    half_size = square_size / 2
                    
                    square_x0 = center_x - half_size
                    square_x1 = center_x + half_size
                    square_y0 = center_y - half_size
                    square_y1 = center_y + half_size
                    
                    # Update zoom state with drawn region (square)
                    current_zoom_state = image_zoom_state.get()
                    new_zoom_state = current_zoom_state.copy()
                    new_zoom_state['drawn_region'] = {
                        'x0': square_x0,
                        'x1': square_x1,
                        'y0': square_y0,
                        'y1': square_y1
                    }
                    new_zoom_state['is_zoomed'] = True
                    image_zoom_state.set(new_zoom_state)
                    print(f"Using drawn square region: x0={square_x0:.1f}, x1={square_x1:.1f}, y0={square_y0:.1f}, y1={square_y1:.1f}")
                else:
                    print("Invalid rectangle coordinates")
            else:
                print(f"Latest shape is not a rectangle: {latest_shape.get('type')}")
        else:
            # No drawn shapes - automatically capture a default region
            print("No drawn shapes - automatically capturing default region")
            
            # Set a default region (center 200x200 pixels)
            default_region = {
                'x0': 400,  # Start at x=400
                'x1': 600,  # End at x=600 (200 pixels wide)
                'y0': 400,  # Start at y=400
                'y1': 600   # End at y=600 (200 pixels high)
            }
            
            # Update zoom state with default region
            current_zoom_state = image_zoom_state.get()
            new_zoom_state = current_zoom_state.copy()
            new_zoom_state['drawn_region'] = default_region
            new_zoom_state['is_zoomed'] = True
            image_zoom_state.set(new_zoom_state)
            
            # Also store in drawn_shapes for consistency
            drawn_shapes.set([{
                'type': 'rect',
                'x0': default_region['x0'],
                'x1': default_region['x1'],
                'y0': default_region['y0'],
                'y1': default_region['y1']
            }])
            
            print(f"Default region captured: {default_region}")
        
        # Trigger FFT calculation
        fft_trigger.set(fft_trigger.get() + 1)
        print("=== END FFT CALCULATION ===")

    @reactive.Effect
    @reactive.event(input.fit_markers)
    def _():
        """Handle Fit button click to fit ellipse to lattice points."""
        current_state = fft_state.get()
        if current_state['mode'] != 'Lattice Point':
            return
            
        points = list(current_state['lattice_points'])
        if len(points) == 0:
            return
            
        # Compute image center
        cx, cy = size / 2, size / 2
        
        # If fewer than 3 points, create mirrored and jittered points
        if len(points) < 3:
            working_points = points.copy()
            for x, y in points:
                # Mirror through center
                mx, my = 2 * cx - x, 2 * cy - y
                # Add small Gaussian noise
                jittered_x = mx + np.random.normal(scale=1.0)
                jittered_y = my + np.random.normal(scale=1.0)
                working_points.append((jittered_x, jittered_y))
        else:
            working_points = points
        
        # Fit ellipse
        try:
            a, b, theta = fit_ellipse_fixed_center(working_points, center=(cx, cy))
            
            # Update state with ellipse parameters
            new_state = current_state.copy()
            new_state['ellipse_params'] = (a, b, theta)
            fft_state.set(new_state)
        except Exception as e:
            print(f"Ellipse fitting failed: {e}")

    @reactive.Effect
    @reactive.event(input.estimate_tilt)
    def _():
        """Handle Estimate Tilt button click to compute tilt angle from ellipse."""
        current_state = fft_state.get()
        if current_state['mode'] != 'Lattice Point':
            return
            
        # Ensure an ellipse is fitted
        if current_state['ellipse_params'] is None:
            points = list(current_state['lattice_points'])
            if len(points) == 0:
                return
                
            # Compute image center
            cx, cy = size / 2, size / 2
            
            # If fewer than 3 points, create mirrored and jittered points
            if len(points) < 3:
                working_points = points.copy()
                for x, y in points:
                    # Mirror through center
                    mx, my = 2 * cx - x, 2 * cy - y
                    # Add small Gaussian noise
                    jittered_x = mx + np.random.normal(scale=1.0)
                    jittered_y = my + np.random.normal(scale=1.0)
                    working_points.append((jittered_x, jittered_y))
            else:
                working_points = points
            
            # Fit ellipse
            try:
                a, b, theta = fit_ellipse_fixed_center(working_points, center=(cx, cy))
                
                # Update state with ellipse parameters
                new_state = current_state.copy()
                new_state['ellipse_params'] = (a, b, theta)
                fft_state.set(new_state)
                current_state = new_state
            except Exception as e:
                print(f"Ellipse fitting failed: {e}")
                return
        
        # Compute tilt from ellipse parameters
        a, b, _ = current_state['ellipse_params']
        small_axis, large_axis = sorted([a, b])
        tilt_angle = calculate_tilt_angle(small_axis, large_axis)
        
        # Calculate apix from large axis (similar to resolution ring mode)
        resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
        if resolution is not None and large_axis > 0:
            # Use the same formula as resolution ring mode: apix = (distance * resolution) / size
            new_apix = (large_axis * resolution) / size
            if 0.01 <= new_apix <= 6.0:
                new_apix = round(new_apix, 3)
                # Update UI controls
                ui.update_slider("apix_slider", value=new_apix, session=session)
                ui.update_text("apix_exact_str", value=str(new_apix), session=session)
        
        # Update state with tilt info
        new_state = current_state.copy()
        new_state['tilt_info'] = (small_axis, large_axis, tilt_angle)
        fft_state.set(new_state)

    # Remove click handler for 1D plot since we're using hover instead of static markers
    
    # Note: Plotly handles zoom and pan automatically, so we don't need separate brush/dblclick handlers
    # The plot_zoom state is still used for programmatic zoom control

    @reactive.Effect
    @reactive.event(input.reset_zoom)
    def _():
        # Reset the zoom state for programmatic control
        plot_zoom.set({'x_range': None, 'y_range': None})
        
        # Update 1D plot widget if it exists
        widget = fft_1d_widget.get()
        if widget is not None:
            # Reset zoom by updating the axis ranges
            with widget.batch_update():
                widget.layout.xaxis.range = None
                widget.layout.yaxis.range = None

    @reactive.Calc
    def image_path():
        file = input.upload()
        if not file:
            return None
        return Path(file[0]["datapath"])

    def save_temp_image(img: Image.Image) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        return tmp.name















    @reactive.Calc
    def get_apix():
        return apix_master.get()

    @reactive.Calc
    def get_apix_from_distance():
        """Calculate the apix value from a given distance in pixels and current resolution.
        
        Returns:
            A function that takes distance in pixels and returns the corresponding apix value.
        """
        resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
        if resolution is None:
            return lambda distance: None
        
        def calculate_apix(distance_pixels):
            """Calculate apix from distance in pixels.
            
            Args:
                distance_pixels: Distance from center in pixels
                
            Returns:
                Apix value in Å/pixel, or None if invalid
            """
            if distance_pixels <= 0:
                return None
            return calculate_apix_from_distance(distance_pixels, resolution, size)
        
        return calculate_apix

    @reactive.Calc
    def get_distance_from_apix():
        """Calculate the distance in pixels from a given apix value and current resolution.
        
        Returns:
            A function that takes apix value and returns the corresponding distance in pixels.
        """
        resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
        if resolution is None:
            return lambda apix: None
        
        def calculate_distance(apix_value):
            """Calculate distance in pixels from apix value.
            
            Args:
                apix_value: Apix value in Å/pixel
                
            Returns:
                Distance from center in pixels, or None if invalid
            """
            if apix_value <= 0:
                return None
            return calculate_distance_from_apix(apix_value, resolution, size)
        
        return calculate_distance



    @reactive.Effect
    @reactive.event(input.upload)
    def _():
        """Update image data when a new file is uploaded."""
        path = image_path()
        if not path or not path.exists():
            raw_image_data.set({'img': None, 'data': None})
            image_data.set(None)
            image_filename.set(None)
            original_image_data.set(None)
            binned_image_data.set(None)
            image_zoom_state.set({'x_range': None, 'y_range': None, 'is_zoomed': False, 'drawn_region': None})
            cached_fft_image.set(None)
            cached_fft_image_no_circles.set(None)
            drawn_region_coords.set(None)
            return
            
        # Load image using the original get_image function (no binning for FFT analysis)
        try:
            print(f"Loading image from: {path}")
            # Load original image without binning for FFT analysis
            original_data, target_apix, original_apix = get_image(str(path))
            print(f"Image loaded successfully: original_shape={original_data.shape}, apix={target_apix}")
            
            # Create binned version for display only
            binned_data = bin_image(original_data, target_size=1000)
            print(f"Created binned version for display: binned_shape={binned_data.shape}")
            
            # Set the binned data for display (always 1000x1000)
            image_data.set(binned_data)
            image_apix.set(target_apix)
            image_filename.set(path.name)
            
            # Store original and binned data separately
            original_image_data.set(original_data)
            binned_image_data.set(binned_data)
            
            # Reset zoom state
            image_zoom_state.set({'x_range': None, 'y_range': None, 'is_zoomed': False, 'drawn_region': None})
            
            # Reset FFT trigger and clear cached FFT images
            fft_trigger.set(0)
            cached_fft_image.set(None)
            cached_fft_image_no_circles.set(None)
            drawn_shapes.set([])
            
            # Also keep the old format for compatibility with FFT calculations
            img = Image.fromarray(binned_data.astype(np.uint8))
            raw_image_data.set({
                'img': img,
                'data': binned_data
            })
            print(f"Image data set successfully")
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
            raw_image_data.set({'img': None, 'data': None})
            image_data.set(None)
            image_filename.set(None)
            original_image_data.set(None)
            binned_image_data.set(None)

    @reactive.Effect
    @reactive.event(input.contrast, fft_trigger)
    def _():
        """Update cached FFT images when contrast changes or FFT is manually triggered."""
        print("FFT calculation triggered - checking drawn region")
        zoom_state = image_zoom_state.get()
        print(f"Current zoom state: {zoom_state}")
        
        region = get_current_region()
        if region is not None:
            print(f"FFT region size: {region.size}")
            # Generate base FFT image without circles
            fft_img = compute_fft_image_region(region, input.contrast())
            cached_fft_image.set(fft_img)
            
            # Generate FFT image without resolution circles for Lattice Point mode
            fft_img_no_circles = fft_img.copy()
            cached_fft_image_no_circles.set(fft_img_no_circles)
            print("FFT images updated successfully")
            
            # Update 1D plot widget if it exists
            widget = fft_1d_widget.get()
            if widget is not None and len(widget.data) > 0:
                # Get updated plot data
                plot_data = fft_1d_data()
                if plot_data is not None:
                    # Update the trace data in-place
                    with widget.batch_update():
                        widget.data[0].x = plot_data['x_data']
                        widget.data[0].y = plot_data['y_data']
                        widget.data[0].name = plot_data['profile_label']
                        
                        # Update y-axis title based on log_y setting
                        if input.log_y():
                            widget.layout.yaxis.title.text = "Log(FFT intensity)"
                        else:
                            widget.layout.yaxis.title.text = "FFT intensity"
        else:
            cached_fft_image.set(None)
            cached_fft_image_no_circles.set(None)
            print("No region available for FFT calculation")

    # Remove the effect that forces FFT redraws - this was causing unnecessary re-renders

    @output
    @render_plotly
    def image_display():
        # Only show the image if it is available
        current_image_data = image_data.get()
        if current_image_data is None:
            # Return None to show nothing when no image is uploaded
            return None
        # Show the image, no title
        fig = plot_image(
            image_data=current_image_data,
            title="",
            apix=image_apix.get(),
            plot_height=None
        )
        
        # Enable drawing interactions (primary mode)
        fig.update_layout(
            dragmode='drawrect',  # Enable rectangle drawing by default
            modebar=dict(
                add=['drawrect', 'eraseshape', 'zoom', 'pan', 'reset+autorange']
            ),
            # Ensure relayout events are captured
            uirevision=None,  # Allow relayout events to be captured
            # Add explicit event handling
            clickmode='event',
            hovermode='closest',
            # Configure drawing parameters for square only
            newshape=dict(
                line_color='cyan',
                line_width=2,
                fillcolor='rgba(0, 255, 255, 0.1)',
                drawdirection='diagonal',
                layer='above'
            ),
            # Explicitly enable shape events
            shapes=[]
        )
        
        # Add annotation to guide users
        # fig.add_annotation(
        #     x=0.5,
        #     y=0.95,
        #     text="Draw a square to select FFT region, then click 'Calc FFT' button",
        #     xref="paper",
        #     yref="paper",
        #     showarrow=False,
        #     font_size=14,
        #     font_color='cyan',
        #     bgcolor='rgba(0, 0, 0, 0.7)',
        #     bordercolor='cyan',
        #     borderwidth=1
        # )
        
        # Ensure axes are configured to send relayout events
        fig.update_xaxes(
            rangeslider_visible=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
        
        return fig

    def get_current_region():
        """Get the current region for FFT calculation based on drawn square or use entire image.
        Uses extract_region_no_binning from compute.py to ensure full-resolution regions for accurate FFT analysis."""
        if image_data.get() is None:
            print("No image data available")
            return None
            
        zoom_state = image_zoom_state.get()
        original_data = original_image_data.get()
        binned_data = binned_image_data.get()
        
        if original_data is None or binned_data is None:
            print("No original or binned data available")
            return None
            
        print(f"Zoom state: {zoom_state}")
        print(f"Original data shape: {original_data.shape}")
        print(f"Binned data shape: {binned_data.shape}")
        
        # Check if there's a drawn region (square)
        if zoom_state.get('drawn_region') is not None:
            drawn_region = zoom_state['drawn_region']
            print(f"Using drawn square region: {drawn_region}")
            
            # Extract region from original image using the compute.py function
            try:
                region_data = extract_region_no_binning(
                    original_data=original_data,
                    binned_data=binned_data,
                    x_range=(drawn_region['x0'], drawn_region['x1']),
                    y_range=(drawn_region['y0'], drawn_region['y1'])
                )
                region_img = Image.fromarray(region_data.astype(np.uint8))
                print(f"Extracted full-resolution region size: {region_img.size} (no binning)")
                return region_img
            except Exception as e:
                print(f"Error extracting drawn region: {e}")
                # Fallback to entire original image
                region_img = Image.fromarray(original_data.astype(np.uint8))
                return region_img
        
        # If no drawn region, use the entire original image (no binning for FFT analysis)
        else:
            # Use the entire original image without binning for FFT calculation
            try:
                region_img = Image.fromarray(original_data.astype(np.uint8))
                print(f"Using entire original image: {region_img.size} (no binning)")
                return region_img
            except Exception as e:
                print(f"Error processing entire image: {e}")
                # Fallback to binned data
                region_img = Image.fromarray(binned_data.astype(np.uint8))
                return region_img

    def get_drawn_shapes_from_figure():
        """Get the current drawn shapes from the image display figure."""
        try:
            # This is a placeholder - in a real implementation, we would need to access the current figure state
            # For now, we'll rely on the stored drawn_region in zoom_state
            return None
        except Exception as e:
            print(f"Error getting drawn shapes: {e}")
            return None

    @output
    @render_plotly
    def fft_with_circle():
        from shiny import req
        req(image_data.get() is not None)
        
        # Check if FFT has been calculated
        cached_fft = cached_fft_image.get()
        if cached_fft is None:
            # Return None to show nothing when no FFT has been calculated
            return None
        
        # Get current FFT markers state (more efficient than full fft_state)
        markers = fft_markers.get()
        
        # Always use the no-circles version for both label modes
        fft_img = cached_fft_image_no_circles.get().copy()
        
        # Convert PIL image to numpy array for Plotly
        fft_arr = np.array(fft_img.convert('L')).astype(np.uint8)

        # Prepare overlays for Plotly
        overlays = {
            'mode': markers['mode'],
            'resolution_click_x': markers['resolution_click_x'],
            'resolution_click_y': markers['resolution_click_y'],
            'lattice_points': markers['lattice_points'],
            'ellipse_params': markers['ellipse_params'],
            'zoom_factor': markers['zoom_factor']
        }

        # Use the imported function to create the Plotly figure
        from compute import create_fft_2d_plotly_figure
        fig = create_fft_2d_plotly_figure(
            fft_data=fft_arr,
            overlays=overlays,
            apix=get_apix(),
            resolution_type=input.resolution_type(),
            custom_resolution=input.custom_resolution(),
            size=fft_arr.shape[0],
            contrast=input.contrast(),
            title=None
        )
        
        # Use the same layout as the original image for consistent styling
        fig.update_layout(
            height=None,  # Allow natural sizing like original image
            margin=dict(l=10, r=10, t=10, b=10),  # Minimal margins
            autosize=True
        )
        
        # Create FigureWidget from the Figure and return it
        fw = FigureWidget(fig)
        return fw

    @reactive.calc
    def fft_1d_data():
        """Calculate the data needed for the 1D FFT plot."""
        from shiny import req
        
        req(image_data.get() is not None)

        region = get_current_region()
        if region is None:
            return None

        return compute_fft_1d_data(
            region=region,
            apix=get_apix(),
            use_mean_profile=input.use_mean_profile(),
            log_y=input.log_y(),
            smooth=input.smooth(),
            window_size=input.window_size(),
            detrend=input.detrend(),
            resolution_type=input.resolution_type(),
            custom_resolution=input.custom_resolution()
        )
    
    @render_plotly("fft_1d_plot")
    def fft_1d_plot():
        # Check if FFT has been calculated
        cached_fft = cached_fft_image.get()
        if cached_fft is None:
            # Return None to show nothing when no FFT has been calculated
            return None
        
        # Get the calculated plot data
        plot_data = fft_1d_data()
        if plot_data is None:
            return go.Figure()
        
        # Get current region and zoom state
        region = get_current_region()
        zoom = plot_zoom.get()
        
        # Get current resolution for apix calculation
        resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
        
        # Create plotly figure using the imported function
        fig = create_fft_1d_plotly_figure(
            plot_data=plot_data,
            resolution=resolution,
            region=region,
            size=size,
            zoom_state=zoom
        )
        
        # Create FigureWidget from the Figure and return it
        fw = FigureWidget(fig)
        
        # Store the widget for in-place updates
        fft_1d_widget.set(fw)
        
        return fw



    @output
    @render.text
    def lattice_points_data():
        """Hidden output to expose lattice points data for persistence."""
        state = fft_state.get()
        if state['mode'] == 'Lattice Point' and state['lattice_points']:
            # Return lattice points as JSON-like string for easy parsing
            points_str = ";".join([f"{x},{y}" for x, y in state['lattice_points']])
            return f"Lattice Points: {points_str}"
        return "Lattice Points: None"

    @output
    @render.text
    def tilt_output():
        """Display tilt estimation results."""
        state = fft_state.get()
        if state['tilt_info'] is not None:
            small_axis, large_axis, tilt_angle = state['tilt_info']
            tilt_angle_degrees = math.degrees(tilt_angle)
            
            # Calculate apix from large axis
            resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
            apix_str = ""
            if resolution is not None and large_axis > 0:
                calculated_apix = (large_axis * resolution) / size
                if 0.01 <= calculated_apix <= 6.0:
                    apix_str = f", Apix: {calculated_apix:.3f} Å/px"
            
            return (f"Small axis: {small_axis:.2f}, "
                   f"Large axis: {large_axis:.2f}, "
                   f"Estimated Tilt Angle: {tilt_angle_degrees:.2f}°"
                   f"{apix_str}")
        return ""

    @output
    @render.text
    def tilt_1d_output():
        """Display 1D tilt estimation results."""
        state = fft_state.get()
        if state['tilt_info_1d'] is not None:
            smaller_x, larger_x, tilt_angle, apix_value = state['tilt_info_1d']
            tilt_angle_degrees = math.degrees(tilt_angle)
            
            return (f"Estimated Tilt Angle: {tilt_angle_degrees:.2f}°, "
                   f"Apix at selected frequency: {apix_value:.3f} Å/px")
        return ""

    # --- All UI controls and plots react to apix_master ---
    @reactive.Effect
    @reactive.event(apix_master)
    def _():
        val = apix_master.get()
        
        # Only update if the change is significant (>= 0.001)
        # This prevents unnecessary updates for tiny changes
        current_slider_val = input.apix_slider()
        if abs(val - current_slider_val) < 0.001:
            return
            
        # Update UI controls
        # ui.update_slider("apix_slider", value=val, session=session)
        # ui.update_text("apix_exact_str", value=str(round(val, 3)), session=session)
        
        # Update FFT circle positions by clearing click positions
        # This will force the circles to use calculated positions instead of clicked positions
        current_state = fft_state.get()
        if current_state['mode'] == 'Resolution Ring':
            new_state = current_state.copy()
            new_state['resolution_radius'] = None
            new_state['resolution_click_x'] = None
            new_state['resolution_click_y'] = None
            fft_state.set(new_state)
        
        # Clear 1D plot clicked position when resolution type changes
        #plot_1d_click_pos.set({'x': None, 'y': None, 'color': None})

    @reactive.Effect
    @reactive.event(input.label_mode)
    def _():
        """Handle mode switching and clear appropriate markers."""
        current_state = fft_state.get()
        new_state = current_state.copy()
        
        if input.label_mode() == "Resolution Ring":
            # Switching to Resolution Ring: clear lattice points, ellipse, tilt info, and 1D plot markers
            new_state['mode'] = 'Resolution Ring'
            new_state['lattice_points'] = []
            new_state['ellipse_params'] = None
            new_state['tilt_info'] = None
            new_state['plot_1d_markers'] = []
            new_state['tilt_info_1d'] = None
        elif input.label_mode() == "Lattice Point":
            # Switching to Lattice Point: clear resolution radius, click coordinates, tilt info, and 1D plot markers
            new_state['mode'] = 'Lattice Point'
            new_state['resolution_radius'] = None
            new_state['resolution_click_x'] = None
            new_state['resolution_click_y'] = None
            new_state['ellipse_params'] = None
            new_state['tilt_info'] = None
            new_state['plot_1d_markers'] = []
            new_state['tilt_info_1d'] = None
        
        fft_state.set(new_state)
        
        # Update Fit button state
        is_disabled = input.label_mode() != "Lattice Point"
        ui.update_action_button("fit_markers", disabled=is_disabled, session=session)
        ui.update_action_button("estimate_tilt", disabled=is_disabled, session=session)
        ui.update_action_button("estimate_tilt_1d", disabled=is_disabled, session=session)

    @reactive.Effect
    @reactive.event(input.zoom2)
    def _():
        """Update zoom factor in fft_state when zoom slider changes."""
        current_state = fft_state.get()
        new_state = current_state.copy()
        new_state['zoom_factor'] = input.zoom2() / 100.0
        fft_state.set(new_state)
    
    # Reactive effects to update 1D plot FigureWidget in-place
    @reactive.Effect
    @reactive.event(input.log_y, input.use_mean_profile, input.smooth, input.detrend, input.window_size)
    def _():
        """Update 1D plot in-place when plot parameters change."""
        widget = fft_1d_widget.get()
        if widget is not None and len(widget.data) > 0:
            # Get updated plot data
            plot_data = fft_1d_data()
            if plot_data is None:
                return
            
            # Update the trace data in-place
            with widget.batch_update():
                widget.data[0].x = plot_data['x_data']
                widget.data[0].y = plot_data['y_data']
                widget.data[0].name = plot_data['profile_label']
                
                # Update y-axis title based on log_y setting
                if input.log_y():
                    widget.layout.yaxis.title.text = "Log(FFT intensity)"
                else:
                    widget.layout.yaxis.title.text = "FFT intensity"

    @reactive.Effect
    @reactive.event(input.estimate_tilt_1d)
    def _():
        """Handle 1D Estimate Tilt button click to compute tilt angle from 1D plot markers."""
        current_state = fft_state.get()
        if current_state['mode'] != 'Lattice Point':
            return
            
        # Check if we have exactly 2 markers
        if len(current_state['plot_1d_markers']) != 2:
            return
            
        # Get the two x positions
        x1 = current_state['plot_1d_markers'][0][0]
        x2 = current_state['plot_1d_markers'][1][0]
        
        # Identify smaller and larger values
        smaller_x, larger_x = sorted([x1, x2])
        
        # Compute tilt angle as arccos(smaller/larger)
        if larger_x > 0:
            tilt_angle = calculate_tilt_angle(smaller_x, larger_x)
        else:
            return
        
        # Convert larger marker position to apix using resolution-to-apix mapping
        resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
        if resolution is not None:
            # Convert from region coordinates to full FFT coordinates
            region = get_current_region()
            if region is not None:
                region_size = region.size[0]
                full_fft_size = size
                fft_radius = larger_x * (full_fft_size / region_size)
                
                # Calculate apix using the same formula as other modes
                apix_value = (fft_radius * resolution) / full_fft_size
                
                if 0.01 <= apix_value <= 6.0:
                    # Update state with 1D tilt info
                    new_state = current_state.copy()
                    new_state['tilt_info_1d'] = (smaller_x, larger_x, tilt_angle, apix_value)
                    fft_state.set(new_state)

    @reactive.Effect
    @reactive.event(input.image_display_relayout)
    def _():
        """Handle shape drawing events for FFT region selection."""
        relayout_data = input.image_display_relayout()
        print(f"=== RELAYOUT EVENT RECEIVED ===")
        print(f"Relayout data: {relayout_data}")
        
        if relayout_data is None:
            print("Relayout data is None")
            return
        
        # Check if this is a shape event (shapes array is present)
        if 'shapes' in relayout_data:
            shapes = relayout_data['shapes']
            print(f"Shapes data: {shapes}")
            
            # Store all shapes in our reactive value
            drawn_shapes.set(shapes)
            print(f"Stored {len(shapes)} shapes in drawn_shapes")
            
            if shapes and len(shapes) > 0:
                # Get the most recent shape (last one in the array)
                latest_shape = shapes[-1]
                print(f"Latest shape: {latest_shape}")
                if latest_shape.get('type') == 'rect':
                    # Extract rectangle coordinates
                    x0 = latest_shape.get('x0')
                    x1 = latest_shape.get('x1')
                    y0 = latest_shape.get('y0')
                    y1 = latest_shape.get('y1')
                    
                    print(f"Rectangle coordinates: x0={x0}, x1={x1}, y0={y0}, y1={y1}")
        
        # Check if shapes were cleared
        elif 'shapes' in relayout_data and relayout_data['shapes'] is None:
            # Clear drawn shapes
            drawn_shapes.set([])
            print("Drawn shapes cleared")
        
        print(f"=== END RELAYOUT EVENT ===")

app = App(app_ui, server)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Magnification Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # We'll handle help manually to use our custom format
    )
    parser.add_argument('--help', '-h', action='store_true', 
                       help='Show detailed help message')

    args = parser.parse_args()
    
    if args.help:
        print_help()
    else:
        app.run()
