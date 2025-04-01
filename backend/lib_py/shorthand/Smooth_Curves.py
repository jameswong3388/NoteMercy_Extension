import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import scipy.interpolate as si
import base64
from io import BytesIO


class StrokeSmoothnessAnalyzer:
    """
    Analyzes the smoothness characteristics of strokes in an image using
    fixed internal parameters. Does not generate runtime warnings.
    """
    def __init__(self, image_input, is_base64=True):
        """Initializes the analyzer by loading the original color image."""
        self.img_color = None
        self.original_height = 0
        self.original_width = 0

        try:
            if is_base64:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                self.img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if self.img_color is None:
                    raise ValueError("Could not decode base64 image as color.")
            else:
                self.img_color = cv2.imread(image_input, cv2.IMREAD_COLOR)
                if self.img_color is None:
                    # Try decoding as base64 as a fallback if path read fails
                    try:
                        img_data = base64.b64decode(image_input)
                        nparr = np.frombuffer(img_data, np.uint8)
                        self.img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if self.img_color is None:
                            raise ValueError("Input is not a valid path or base64 string.")
                    except Exception as e_inner:
                        raise ValueError(
                            f"Could not read image at path '{image_input}' or decode as base64. Error: {e_inner}")

        except Exception as e:
            # Propagate critical loading errors
            raise ValueError(f"Error loading image: {e}")

        if self.img_color is not None:
            self.original_height, self.original_width = self.img_color.shape[:2]
        else:
            # This case should be caught by exceptions above, but as a failsafe:
            raise ValueError("Image could not be loaded successfully.")

        self._reset_analysis_data()

    def _reset_analysis_data(self):
        """Clears intermediate data from previous analysis runs."""
        self.gray_image = None
        self.binary_image = None
        self.ordered_points = None
        self.spline_data = None
        self.analysis_metrics = {}
        self.analysis_plot_data = {}
        # self.analysis_warnings = [] # Removed

    def preprocess_image(self):
        """
        Applies fixed preprocessing: Grayscale, Gaussian Blur, Simple Thresholding.
        Stores the final binary result (white strokes=255, black background=0)
        in self.binary_image.
        """
        if self.img_color is None:
            # Cannot proceed without an image loaded during init
            self.binary_image = None
            return

        # --- Fixed Preprocessing Parameters ---
        _GAUSSIAN_BLUR_KSIZE = 3
        _THRESH_VALUE = 127
        _THRESH_MAX_VALUE = 255
        _THRESH_TYPE = cv2.THRESH_BINARY_INV
        # --- End Parameters ---

        # 1. Grayscale Conversion
        if len(self.img_color.shape) == 3 and self.img_color.shape[2] == 3:
            gray_image = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        elif len(self.img_color.shape) == 2:
            gray_image = self.img_color.copy()  # Already grayscale
        else:
            gray_image = cv2.cvtColor(self.img_color, cv2.COLOR_BGRA2GRAY)

        processed = gray_image
        self.gray_image = gray_image

        # 2. Gaussian Blur
        if _GAUSSIAN_BLUR_KSIZE > 1:
            # Ensure kernel size is odd
            ksize = _GAUSSIAN_BLUR_KSIZE if _GAUSSIAN_BLUR_KSIZE % 2 != 0 else _GAUSSIAN_BLUR_KSIZE + 1
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

        # 3. Simple Thresholding
        ret, thresholded_image = cv2.threshold(processed, _THRESH_VALUE, _THRESH_MAX_VALUE, _THRESH_TYPE)
        self.binary_image = thresholded_image

    def _calculate_smoothness_metrics(self):
        """
        Calculates stroke smoothness metrics. Performs skeletonization, sorts points,
        fits spline, and calculates metrics. Returns early with default metrics
        if critical steps fail (e.g., insufficient points, spline error).
        """
        # Fixed parameters for analysis
        min_points_for_spline = 10  # Need at least k+1 points for spline
        spline_degree = 3  # Cubic spline
        spline_smoothing_factor = 0  # Interpolate through points
        spline_interp_factor = 2  # Evaluate spline at 2x ordered points (min 200)

        # Initialize default metrics
        self.analysis_metrics = {
            'avg_abs_angle_change': 0.0, 'angle_change_variance': 0.0,
            'normalized_direction_changes': 0.0, 'stroke_length': 0.0,
            'num_skeleton_points': 0, 'num_ordered_points': 0
        }
        # Initialize plot data structure
        self.analysis_plot_data = {'skeleton_image': None, 'skeleton_points': None,
                                   'ordered_points': None, 'spline_points': None,
                                   'spline_param': None, 'angle_changes': None}

        if self.binary_image is None:
            # Cannot proceed if preprocessing failed or produced no result
            return

        # --- Perform Skeletonization ---
        skeleton_image_uint8 = None  # Define scope
        try:
            binary_bool = self.binary_image > 0
            if not np.any(binary_bool):
                # Image is empty after thresholding, nothing to skeletonize
                return  # Return with default (zero) metrics

            skel_bool = skeletonize(binary_bool)
            skeleton_image_uint8 = skel_bool.astype(np.uint8) * 255
            self.analysis_plot_data['skeleton_image'] = skeleton_image_uint8
        except Exception:
            # Cannot proceed if skeletonization itself fails unexpectedly
            # print(f"Debug: Skeletonization failed: {e}") # Optional debug print
            return

        # If skeletonization produced an empty image (e.g., input was just noise removed by skeletonize)
        if skeleton_image_uint8 is None or not np.any(skeleton_image_uint8):
            return  # Return with default (zero) metrics

        # 1. Extract Skeleton Points
        ys, xs = np.nonzero(skeleton_image_uint8)
        num_skeleton_points = len(xs)
        self.analysis_metrics['num_skeleton_points'] = num_skeleton_points
        # Store as (x, y) pairs
        raw_skeleton_points = np.column_stack((xs, ys))
        self.analysis_plot_data['skeleton_points'] = raw_skeleton_points

        # Need at least 2 points to form a path/line
        if num_skeleton_points < 2:
            return  # Not enough points for any analysis

        # 2. Sort Skeleton Points
        # Note: The basic sorting can still be unreliable for complex strokes.
        self.ordered_points = self._sort_skeleton_points_basic(raw_skeleton_points)
        num_ordered_points = len(self.ordered_points)
        self.analysis_metrics['num_ordered_points'] = num_ordered_points
        self.analysis_plot_data['ordered_points'] = self.ordered_points

        # Need enough points for spline fitting (k+1)
        min_pts_needed = spline_degree + 1
        if num_ordered_points < min_pts_needed or num_ordered_points < min_points_for_spline:
            # Silently return if not enough points after sorting
            # print(f"Debug: Not enough ordered points ({num_ordered_points}) for spline (k={spline_degree}). Min required: {max(min_pts_needed, min_points_for_spline)}") # Optional
            return

        # 3. Spline Fitting
        x_sorted = self.ordered_points[:, 0]
        y_sorted = self.ordered_points[:, 1]
        try:
            # Check for sufficient unique points along axes to avoid rank deficiency in splprep
            # This check prevents errors like: "error: The number of derivatives at boundaries does not match the number of coordinates"
            if len(np.unique(x_sorted)) < min_pts_needed or len(np.unique(y_sorted)) < min_pts_needed:
                # Fallback for degenerate cases (e.g., perfectly straight lines)
                # Could potentially try a lower degree spline, but for now, just skip spline part.
                # print(f"Debug: Insufficient unique points for spline fitting (unique X: {len(np.unique(x_sorted))}, unique Y: {len(np.unique(y_sorted))})") # Optional
                return  # Skip spline fitting and metric calculation

            tck, u = si.splprep([x_sorted, y_sorted], s=spline_smoothing_factor, k=spline_degree, quiet=True)
            num_fine_points = max(200, num_ordered_points * spline_interp_factor)
            u_fine = np.linspace(u.min(), u.max(), num=num_fine_points)  # Use range from u
            spline_x, spline_y = si.splev(u_fine, tck)
            self.spline_data = {'tck': tck, 'u': u, 'u_fine': u_fine, 'x': spline_x, 'y': spline_y}
            self.analysis_plot_data['spline_points'] = np.column_stack((spline_x, spline_y))
            self.analysis_plot_data['spline_param'] = u_fine
        except (TypeError, ValueError, np.linalg.LinAlgError) as e:
            # Catch common spline fitting errors silently
            # print(f"Debug: Spline fitting failed: {e}") # Optional debug print
            # Metrics will remain at default values
            return

        # 4. Metric Calculation (only if spline fitting succeeded)
        try:
            spline_x = self.spline_data['x']
            spline_y = self.spline_data['y']
            # Ensure enough points for gradient/diff
            if len(spline_x) < 2: return

            dx = np.gradient(spline_x)
            dy = np.gradient(spline_y)
            # Use atan2(dy, dx) for standard angle calculation relative to +X axis
            # Note: Y in image is typically down, so dy might be 'inverted' compared
            # to standard math plots, but atan2 handles quadrants correctly.
            # If using angles for *change*, the relative difference is key.
            theta_rad = np.arctan2(dy, dx)
            theta_unwrapped = np.unwrap(theta_rad)  # Handles jumps from -pi to +pi

            # Ensure enough points for diff
            if len(theta_unwrapped) < 2: return

            dTheta = np.diff(theta_unwrapped)  # Angle change between consecutive points
            self.analysis_plot_data['angle_changes'] = dTheta

            segment_lengths_px = np.sqrt(np.diff(spline_x) ** 2 + np.diff(spline_y) ** 2)
            stroke_length = np.sum(segment_lengths_px)
            safe_stroke_length = max(stroke_length, 1e-6)  # Avoid division by zero

            avg_abs_angle_change = np.mean(np.abs(dTheta)) if len(dTheta) > 0 else 0.0
            angle_change_variance = np.var(dTheta) if len(dTheta) > 0 else 0.0

            # Direction changes: Count sign flips in angle changes (curvature direction changes)
            # Filter near-zero changes first to avoid noise sensitivity
            significant_angle_changes = dTheta[np.abs(dTheta) > 1e-6]
            if len(significant_angle_changes) > 1:
                sign_diff = np.diff(np.sign(significant_angle_changes))
                # Count where sign difference is non-zero (approx +/- 2)
                num_direction_changes = np.sum(np.abs(sign_diff) > 1e-6)
            else:
                num_direction_changes = 0
            normalized_direction_changes = num_direction_changes / safe_stroke_length

            # Update metrics only if calculation succeeds
            self.analysis_metrics.update({
                'avg_abs_angle_change': float(avg_abs_angle_change),
                'angle_change_variance': float(angle_change_variance),
                'normalized_direction_changes': float(normalized_direction_changes),
                'stroke_length': float(stroke_length),
            })
        except Exception:
            # Silently return if metric calculation fails
            # print(f"Debug: Metric calculation failed: {e}") # Optional debug print
            # Metrics will retain default values
            return

    @staticmethod
    def _sort_skeleton_points_basic(points_xy):
        """Basic skeleton point sorting using Nearest Neighbor."""
        if points_xy is None or len(points_xy) < 2:
            return points_xy if points_xy is not None else np.array([])

        num_points = len(points_xy)
        ordered_points_list = []
        # Work with a copy to avoid modifying the original array if passed by reference elsewhere
        remaining_points = points_xy.tolist()

        # Start from a consistent point, e.g., top-most then left-most
        # Find index of point with min y, then min x among those with min y
        min_y = min(p[1] for p in remaining_points)
        start_candidates_indices = [i for i, p in enumerate(remaining_points) if p[1] == min_y]
        start_idx = min(start_candidates_indices, key=lambda i: remaining_points[i][0])
        current_point = remaining_points.pop(start_idx)
        ordered_points_list.append(current_point)

        # Convert remaining to NumPy array once for efficient distance calculation
        remaining_np = np.array(remaining_points)

        while len(ordered_points_list) < num_points:
            if remaining_np.size == 0: break  # Should not happen if logic is correct, but safety check

            current_np_point = np.array(current_point)
            # Calculate Euclidean distances from the current point to all remaining points
            distances = np.linalg.norm(remaining_np - current_np_point, axis=1)

            # Find the index of the nearest point *within the remaining_np array*
            nearest_idx_in_remaining = np.argmin(distances)

            # Get the actual point coordinates
            current_point = remaining_np[nearest_idx_in_remaining].tolist()
            ordered_points_list.append(current_point)

            # Remove the found point from remaining_np for the next iteration
            remaining_np = np.delete(remaining_np, nearest_idx_in_remaining, axis=0)

        return np.array(ordered_points_list)

    def _generate_visualization(self):
        """Generates visualization plots using fixed internal display parameters."""
        # --- Fixed Plotting Parameters ---
        figure_size = (12, 8)
        layout_padding = 1.5
        skeleton_plot_color = 'gray'
        spline_plot_color = 'b-'  # Blue line for spline
        spline_plot_linewidth = 1.5
        metrics_text_fontsize = 9
        metrics_text_family = 'monospace'
        # --- End Parameters ---

        graphs = []
        # Check essential data needed for any plotting
        if self.img_color is None or self.binary_image is None:
            # Cannot plot without original or binary images
            return graphs
        # Plot data might be partially filled if analysis stopped early
        metrics = self.analysis_metrics
        plot_data = self.analysis_plot_data

        # Safely get plot-specific data using .get()
        num_skel_pts = metrics.get('num_skeleton_points', 0)
        num_ord_pts = metrics.get('num_ordered_points', 0)
        skeleton_image_to_plot = plot_data.get('skeleton_image')  # This is the background for plots 3 & 4
        spline_points = plot_data.get('spline_points')
        angle_changes = plot_data.get('angle_changes')
        spline_param = plot_data.get('spline_param')

        try:
            plt.figure("Stroke Smoothness Analysis", figsize=figure_size)

            # --- Plot 1: Original Image ---
            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB))
            plt.title("Original Image");
            plt.axis('off')

            # --- Plot 2: Preprocessed (Binary) Image ---
            plt.subplot(2, 3, 2)
            # Use binary_image directly, which should have white stroke on black bg
            plt.imshow(self.binary_image, cmap='gray')
            plt.title("Preprocessed (Binary)");
            plt.axis('off')

            # --- Plot 3: Skeleton Image ---
            ax3 = plt.subplot(2, 3, 3)
            title_skel = f"Skeleton ({num_skel_pts} pts)"
            if skeleton_image_to_plot is not None:
                # imshow sets origin top-left by default for images
                ax3.imshow(skeleton_image_to_plot, cmap=skeleton_plot_color)
                if num_ord_pts != num_skel_pts and num_ord_pts > 0:
                    title_skel += f"\n(Sorted: {num_ord_pts} pts)"
                ax3.set_aspect('equal', adjustable='box')  # Maintain aspect ratio
            else:
                ax3.text(0.5, 0.5, "Skeleton N/A", ha='center', va='center', transform=ax3.transAxes)
            # No need to invert Y axis when using imshow
            ax3.set_title(title_skel);
            ax3.axis('off')

            # --- Plot 4: Fitted Spline on Skeleton ---
            ax4 = plt.subplot(2, 3, 4)
            title_spline = "Fitted Spline"
            if skeleton_image_to_plot is not None:
                # Display skeleton background first, let imshow set axes extent/orientation
                ax4.imshow(skeleton_image_to_plot, cmap=skeleton_plot_color)
                # Set aspect ratio after imshow
                ax4.set_aspect('equal', adjustable='box')
                # Plot spline coordinates onto the axes established by imshow
                if spline_points is not None and len(spline_points) > 0:
                    ax4.plot(spline_points[:, 0], spline_points[:, 1], spline_plot_color,
                             linewidth=spline_plot_linewidth)
                else:
                    # Only add N/A text if spline failed *after* skeletonization worked
                    title_spline += " (N/A)"
            else:
                # If skeleton itself wasn't available
                ax4.text(0.5, 0.5, "Skeleton/Spline N/A", ha='center', va='center', transform=ax4.transAxes)
                title_spline = "Skeleton/Spline N/A"

            # No need to invert Y axis here either
            ax4.set_title(title_spline);
            ax4.axis('off')

            # --- Plot 5: Angle Change Plot ---
            ax5 = plt.subplot(2, 3, 5)
            title_angle = "Angle Change (deg)"
            # Check if angle change data is available and meaningful
            if angle_changes is not None and spline_param is not None and len(angle_changes) > 0 and len(
                    spline_param) == len(angle_changes) + 1:
                avg_abs_deg = np.degrees(metrics.get('avg_abs_angle_change', 0))
                # Variance is in rad^2, convert to deg^2 for consistency (though unit is awkward)
                var_rad_sq = metrics.get('angle_change_variance', 0)
                std_dev_rad = np.sqrt(var_rad_sq) if var_rad_sq >= 0 else 0
                std_dev_deg = np.degrees(std_dev_rad)
                # Plot angle change (in degrees) vs the spline parameter u (skipping first u value)
                ax5.plot(spline_param[1:], np.degrees(angle_changes))
                # ax5.set_title(f"{title_angle}\nAvgAbs: {avg_abs_deg:.2f}, StdDev: {std_dev_deg:.2f} deg") # Using Std Dev might be more intuitive than Var
                ax5.set_title(f"{title_angle}\nAvgAbs: {avg_abs_deg:.2f} deg")  # Simpler title
                ax5.set_xlabel("Spline Parameter (u)");
                ax5.set_ylabel("Angle Change (deg)")
                # Optional: Add horizontal line at 0 degrees change
                ax5.axhline(0, color='red', linestyle='--', linewidth=0.5)
            else:
                # If data is missing or inconsistent
                ax5.set_title(f"{title_angle}\n(No data)")
                ax5.text(0.5, 0.5, "Angle Change Data N/A", ha='center', va='center', transform=ax5.transAxes)
                ax5.set_xlabel("Spline Parameter (u)");
                ax5.set_ylabel("Angle Change (deg)")

            # --- Plot 6: Metrics Text ---
            ax6 = plt.subplot(2, 3, 6)
            metrics_text_lines = ["Metrics:"]
            # Display metrics that were calculated
            metric_labels = {
                'num_skeleton_points': "Skeleton Points",
                'num_ordered_points': "Ordered Points",
                'stroke_length': "Stroke Length (px)",
                'avg_abs_angle_change': "Avg Abs Angle Chg (rad)",
                'angle_change_variance': "Angle Chg Var (rad²)",
                'normalized_direction_changes': "Norm. Dir. Changes (/px)"
            }
            for key, label in metric_labels.items():
                value = metrics.get(key)  # Use .get() in case a metric wasn't calculated
                if value is not None:
                    if isinstance(value, float):
                        metrics_text_lines.append(f"- {label}: {value:.4f}")
                    else:
                        metrics_text_lines.append(f"- {label}: {value}")
                # else: metrics_text_lines.append(f"- {label}: N/A") # Optionally show N/A

            metrics_text = "\n".join(metrics_text_lines)
            # Use axes coordinates (0 to 1) for text placement
            ax6.text(0.05, 0.95, metrics_text, ha='left', va='top', fontsize=metrics_text_fontsize,
                     family=metrics_text_family, wrap=True, transform=ax6.transAxes)
            ax6.axis('off');
            ax6.set_title("Analysis Summary")

            # --- Finalize Plot Layout ---
            plt.tight_layout(pad=layout_padding)
            buf = BytesIO()
            # Save with sufficient DPI for clarity if viewed larger
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            # IMPORTANT: Close the figure to free memory
            plt.close()
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            graphs.append(plot_base64)

        except Exception as plot_err:
            # Silently ignore plotting errors if they occur
            # print(f"Error during plot generation: {plot_err}") # Optionally print for debugging
            try:
                plt.close()  # Ensure figure is closed even if error occurs
            except Exception:
                pass

        return graphs

    def analyze(self, debug=False):
        """Orchestrates the stroke smoothness analysis process."""
        self._reset_analysis_data()
        self.preprocess_image()

        # Only proceed to metric calculation if preprocessing was successful
        if self.binary_image is not None:
            self._calculate_smoothness_metrics()
        # If preprocessing failed, metrics will remain at default (zero) values

        # Prepare result structure
        result = {
            'metrics': self.analysis_metrics,
            'graphs': [],
            'preprocessed_image': None
        }

        # Generate visualization if requested
        if debug:
            # Attempt visualization even if metrics calculation failed partially
            # (e.g., shows skeleton if spline failed)
            result['graphs'] = self._generate_visualization()
            # No warning printed if graphs list is empty

        # Encode preprocessed image if it exists
        if self.binary_image is not None:
            try:
                is_success, buffer = cv2.imencode('.png', self.binary_image)
                if is_success:
                    result['preprocessed_image'] = base64.b64encode(buffer).decode('utf-8')
                # else: # Silently ignore encoding failure
                #    print("Debug: Failed to encode preprocessed image.") # Optional
            except Exception:
                # Silently ignore any other encoding errors
                pass

        return result


# === Example Usage (No change needed, but output will not show warnings) ===
if __name__ == "__main__":
    image_path = r"C:\Users\Samson\Desktop\Coding\IPPR\backend\atest\calligraphic.png"
    analyzer = StrokeSmoothnessAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)

    print("\n===== Stroke Smoothness Analysis Results =====")
    metrics = results['metrics']
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Display the image directly without saving
    if results['graphs']:
        from PIL import Image
        import io

        print("\nDisplaying visualization...")
        img_data = base64.b64decode(results['graphs'][0])
        img = Image.open(io.BytesIO(img_data))
        img.show()
