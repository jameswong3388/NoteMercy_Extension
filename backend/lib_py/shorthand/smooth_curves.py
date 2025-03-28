import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
# Use adaptive thresholding from OpenCV
# from skimage.filters import threshold_otsu # Replaced
import scipy.interpolate as si
import base64
from io import BytesIO
import warnings

# Optional: Install skan for better skeleton analysis
# pip install skan
# from skan import csr # Example import

class StrokeSmoothnessAnalyzer:
    def __init__(self, image_input, is_base64=True):
        if is_base64:
            try:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if self.img is None:
                    raise ValueError("Error: Could not decode base64 image")
            except Exception as e:
                 raise ValueError(f"Error processing base64 image: {e}")
        else:
            self.img = cv2.imread(image_input)
            if self.img is None:
                raise ValueError(f"Error: Could not read image at {image_input}")

        if len(self.img.shape) == 3:
            self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_img = self.img.copy()

        # Basic noise reduction
        self.gray_img = cv2.GaussianBlur(self.gray_img, (3, 3), 0)

    @staticmethod
    def _sort_skeleton_points_basic(points):
        """
        Basic skeleton point sorting (Nearest Neighbor).
        WARNING: This method is inaccurate for complex strokes, loops, or intersections.
                 Consider using graph-based traversal (e.g., with the 'skan' library)
                 for robust results.
        """
        if len(points) == 0:
            return points

        ordered_points = []
        remaining_points = points.tolist()

        # Start with the leftmost point
        current_idx = min(range(len(remaining_points)), key=lambda i: remaining_points[i][0])
        current_point = remaining_points.pop(current_idx)
        ordered_points.append(current_point)

        while remaining_points:
            current_np = np.array(current_point)
            remaining_np = np.array(remaining_points)
            distances = np.linalg.norm(remaining_np - current_np, axis=1)

            # Find nearest *unvisited* point
            nearest_idx = np.argmin(distances)
            # Add threshold? Prevent large jumps? Basic NN is prone to errors.
            # Simple nearest neighbor:
            current_point = remaining_points.pop(nearest_idx)
            ordered_points.append(current_point)

        return np.array(ordered_points)

    def analyze(self, debug=False, adaptive_block_size=11, adaptive_C=5, min_points_for_spline=10):
        """
        Analyzes the image to determine stroke smoothness characteristics.

        Parameters:
            debug (bool): If True, generates visualization plots.
            adaptive_block_size (int): Size of the neighborhood area for adaptive thresholding (must be odd).
            adaptive_C (int): Constant subtracted from the mean in adaptive thresholding.
            min_points_for_spline (int): Minimum number of skeleton points required for spline fitting.


        Returns:
            dict: Metrics including curvature information, plus visualization graphs if debug=True.
                  Returns default zero metrics if processing fails or insufficient points.
        """
        default_metrics = {
            'avg_abs_angle_change': 0,
            'angle_change_variance': 0,
            'normalized_direction_changes': 0, # Original metric, potentially sensitive
            'stroke_length': 0,
            'num_skeleton_points': 0
        }
        default_result = {'metrics': default_metrics, 'graphs': []}

        # --- Preprocessing ---
        # Adaptive Binarization (more robust to lighting changes)
        binary_inv = cv2.adaptiveThreshold(self.gray_img, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           adaptive_block_size, adaptive_C)
        # Convert to boolean for skeletonize (True for foreground)
        binary = binary_inv > 0

        # --- Skeletonization ---
        skel = skeletonize(binary)
        skel_display = skel.astype(np.uint8) * 255 # For visualization

        # Find skeleton points (note: np.nonzero returns (y, x) indices)
        ys, xs = np.nonzero(skel)
        num_skeleton_points = len(xs)

        if num_skeleton_points < min_points_for_spline:
            warnings.warn(f"Insufficient skeleton points ({num_skeleton_points}) found. Need at least {min_points_for_spline}.")
            metrics = default_metrics.copy()
            metrics['num_skeleton_points'] = num_skeleton_points
            # Add skeleton plot even if analysis fails
            if debug:
                plt.figure("Stroke Smoothness Analysis (Partial)", figsize=(5, 5))
                plt.imshow(skel_display, cmap='gray')
                plt.title(f"Skeleton ({num_skeleton_points} points - Too Few)")
                plt.axis('off')
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                default_result['graphs'] = [plot_base64]

            default_result['metrics'] = metrics
            return default_result


        # --- Point Ordering (CRITICAL STEP - using basic sort with warning) ---
        points = np.column_stack((xs, ys))
        # WARNING: This basic sort is unreliable for complex strokes! Use 'skan' if possible.
        ordered_points = self._sort_skeleton_points_basic(points)
        if len(ordered_points) != num_skeleton_points:
             warnings.warn("Point sorting might have lost points, indicating potential issues.")

        if len(ordered_points) < min_points_for_spline:
             warnings.warn(f"Insufficient ordered points ({len(ordered_points)}) after sorting.")
             metrics = default_metrics.copy()
             metrics['num_skeleton_points'] = num_skeleton_points # Report original count
             # Potentially add skeleton plot here too if desired
             default_result['metrics'] = metrics
             return default_result

        x_sorted = ordered_points[:, 0]
        y_sorted = ordered_points[:, 1]

        # --- Spline Fitting ---
        try:
            # Consider experimenting with s > 0 for smoothing, e.g., s=len(x_sorted)
            tck, u = si.splprep([x_sorted, y_sorted], s=0, k=3) # k=3 for cubic spline
            # Increase number of points for finer gradient calculation
            u_fine = np.linspace(0, 1, num=max(200, num_skeleton_points * 2))
            spline_x, spline_y = si.splev(u_fine, tck)
        except Exception as e:
            warnings.warn(f"Error during spline fitting: {e}")
            # Optionally capture skeleton plot here before returning
            metrics = default_metrics.copy()
            metrics['num_skeleton_points'] = num_skeleton_points
            default_result['metrics'] = metrics
            return default_result

        # --- Metric Calculation ---
        # Calculate gradients and angles
        dx = np.gradient(spline_x)
        dy = np.gradient(spline_y)
        segment_lengths = np.sqrt(dx**2 + dy**2)

        # Avoid division by zero for zero-length segments (shouldn't happen with splprep)
        segment_lengths[segment_lengths == 0] = 1e-6

        # Angles (in radians)
        theta_rad = np.arctan2(dy, dx)
        theta_unwrapped = np.unwrap(theta_rad) # Handle angle wrapping (e.g., 359 deg -> 1 deg)

        # Change in angle between consecutive points on the spline
        dTheta = np.diff(theta_unwrapped)

        # Stroke length (sum of distances between interpolated points)
        stroke_length = np.sum(np.sqrt(np.diff(spline_x)**2 + np.diff(spline_y)**2))
        if stroke_length < 1e-6: stroke_length = 1e-6 # Avoid division by zero

        # Average absolute change in angle (related to average curvature)
        avg_abs_angle_change = np.mean(np.abs(dTheta)) if len(dTheta) > 0 else 0

        # Variance of angle change (how much curvature fluctuates)
        angle_change_variance = np.var(dTheta) if len(dTheta) > 0 else 0

        # Original direction changes metric (count sign changes in dTheta)
        # Consider replacing/supplementing with a count of changes > threshold
        direction_sign_changes = np.sum(np.abs(np.diff(np.sign(dTheta))) > 1e-6) if len(dTheta) > 1 else 0
        normalized_direction_changes = direction_sign_changes / stroke_length


        # Store metrics (removed combined score)
        metrics = {
            'avg_abs_angle_change': avg_abs_angle_change,
            'angle_change_variance': angle_change_variance,
            'normalized_direction_changes': normalized_direction_changes,
            'stroke_length': stroke_length,
            'num_skeleton_points': num_skeleton_points
        }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # --- Debug Visualization ---
        if debug:
            try:
                plt.figure(f"Stroke Smoothness Analysis", figsize=(12, 8))

                # Original image
                plt.subplot(2, 3, 1)
                plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
                plt.title("Original Image")
                plt.axis('off')

                # Binarized Image
                plt.subplot(2, 3, 2)
                plt.imshow(binary_inv, cmap='gray')
                plt.title("Adaptive Threshold")
                plt.axis('off')

                # Skeleton image
                plt.subplot(2, 3, 3)
                plt.imshow(skel_display, cmap='gray')
                # Overlay sorted points to visualize order (can be messy)
                # plt.plot(x_sorted, y_sorted, 'r.-', markersize=2, linewidth=0.5)
                plt.title(f"Skeleton ({num_skeleton_points} points)")
                plt.axis('off')

                # Fitted spline on Skeleton
                plt.subplot(2, 3, 4)
                plt.imshow(skel_display, cmap='gray')
                plt.plot(spline_x, spline_y, 'b-', linewidth=1.5)
                # Ensure correct aspect ratio and flip y-axis to match image coordinates
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title("Fitted Spline")
                plt.axis('off')


                # Angle Change Plot
                plt.subplot(2, 3, 5)
                if len(dTheta) > 0:
                    plt.plot(u_fine[1:], np.degrees(dTheta)) # Plot against spline parameter u
                    plt.title(f"Angle Change (deg)\nAvgAbs: {np.degrees(avg_abs_angle_change):.2f}, Var: {np.degrees(angle_change_variance):.2f}")
                    plt.xlabel("Spline Parameter")
                    plt.ylabel("Angle Change (deg)")
                else:
                    plt.title("Angle Change (No data)")


                # Metrics Text
                plt.subplot(2, 3, 6)
                metrics_text = (
                    f"Metrics:\n"
                    f"Avg Abs Angle Change: {avg_abs_angle_change:.4f} rad\n"
                    f"Angle Change Variance: {angle_change_variance:.4f}\n"
                    f"Norm. Dir Changes: {normalized_direction_changes:.4f}\n"
                    f"Stroke Length: {stroke_length:.2f} px\n"
                    f"Skeleton Points: {num_skeleton_points}"
                )
                plt.text(0.05, 0.95, metrics_text, ha='left', va='top', fontsize=9, family='monospace')
                plt.axis('off')
                plt.title("Calculated Metrics")


                plt.tight_layout()

                # Convert plot to base64
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.show()
                plt.close()

                result['graphs'].append(plot_base64)

            except Exception as plot_err:
                warnings.warn(f"Error during debug plot generation: {plot_err}")
                # Ensure figure is closed if error occurs mid-plot
                plt.close()


        return result


# === Example usage ===
if __name__ == '__main__':
    # Example with file path - Replace with a path to your shorthand word image
    # Use a CLEAR image of a SINGLE word for best results with this basic implementation.
    image_path = '../../atest/shorthand2.png' # <--- CHANGE THIS

    try:
        # Adjust adaptive threshold parameters if needed based on image characteristics
        analyzer = StrokeSmoothnessAnalyzer(image_path, is_base64=False)
        results = analyzer.analyze(debug=True, adaptive_block_size=25, adaptive_C=10)

        print("\n--- Analysis Metrics ---")
        for key, value in results['metrics'].items():
            print(f"{key}: {value:.4f}")

        if results['graphs']:
            print("\nDebug graph generated (base64 encoded).")
            # You can save or display the base64 graph if needed
            # with open("debug_graph.png", "wb") as f:
            #     f.write(base64.b64decode(results['graphs'][0]))
        else:
             print("\nNo debug graph generated.")

    except ValueError as e:
        print(f"Initialization Error: {e}")
    except FileNotFoundError:
         print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")