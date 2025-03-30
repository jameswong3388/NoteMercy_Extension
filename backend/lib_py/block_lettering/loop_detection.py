import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class LoopDetectionAnalyzer:
    """
    Analyzes an image of handwriting to detect and quantify loops
    characteristic of cursive or mixed styles, attempting to exclude
    structural holes found in standard block letters. Uses fixed
    internal parameters for processing and filtering.
    """
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer by loading the image.

        Parameters:
            image_input (str): Base64 encoded image string or image file path.
            is_base64 (bool): True if image_input is base64, False if it's a path.
        """
        self.img = None
        if is_base64:
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            # Decode as color
            self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if self.img is None: raise ValueError("Could not decode base64 image.")
        else:
            # Read as color
            self.img = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if self.img is None: raise ValueError(f"Could not read image at path: {image_input}")

        # Get dimensions from the loaded image
        self.original_height, self.original_width = self.img.shape[:2]

        # --- Internal state (results of preprocessing/analysis) ---
        self.gray_image = None  # Will be created in _preprocess_image
        self.binary_image = None  # Will be created in _preprocess_image
        self.detected_loops = []
        self.loop_vertex_counts = []
        self.loop_areas = []
        self.analyzed_top_level_shapes = 0
        self.shapes_with_loops = 0

    def _reset_analysis_data(self):
        """
        Resets lists that store results between runs (if object is reused).
        """
        self.detected_loops = []
        self.loop_vertex_counts = []
        self.loop_areas = []
        self.analyzed_top_level_shapes = 0
        self.shapes_with_loops = 0

    def _preprocess_image(self):
        """
        Applies fixed preprocessing steps.
        """
        # 1. Convert to Grayscale (Moved from __init__)
        self.gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # 2. Apply fixed Gaussian blur kernel size
        BLUR_KSIZE = 25
        # Use the newly created gray_image for blurring
        processed = cv2.GaussianBlur(self.gray_image, (BLUR_KSIZE, BLUR_KSIZE), 0)

        # 3. Fixed Otsu's thresholding
        _, self.binary_image = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Morphological closing
        kernel = np.ones((3, 3), np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel)

    def _find_contours_and_filter_loops(self):
        """
        Finds contours, filters out tiny noise contours (like dots),
        and then applies fixed filtering rules (area, vertex count)
        to identify style loops based on hierarchy.
        """
        # --- Fixed Filtering Parameters ---
        MIN_CONTOUR_AREA_OVERALL = 200  # NEW: Min area to consider ANY contour (filters dots)
        LOOP_MIN_AREA = 15             # Min area of the inner hole
        PARENT_MIN_AREA = 50           # Min area of the outer shape containing the hole
        APPROX_EPSILON_FACTOR = 0.025  # Factor for approxPolyDP accuracy
        MAX_LOOP_VERTICES = 6          # Max corners allowed for a smooth loop

        # 1. Find all contours initially
        all_contours_initial, hierarchy_initial = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 2. Filter out very small contours (dots, noise)
        valid_indices = set()
        filtered_contours = []
        for i, contour in enumerate(all_contours_initial):
            area = cv2.contourArea(contour)
            # Keep contour only if its area meets the *overall* minimum
            if area >= MIN_CONTOUR_AREA_OVERALL:
                valid_indices.add(i)
                filtered_contours.append(contour)  # Keep the contour itself if needed later, but we mostly need indices

        # Use the original hierarchy but process only valid indices
        all_hierarchy = hierarchy_initial[0]

        # Clear previous results before populating based on filtered contours
        self._reset_analysis_data()

        parent_indices_with_loops = set() # Track parents containing valid loops

        # 3. Iterate through the *original* indices, but check if they are still valid
        for i in range(len(all_hierarchy)):
            # Skip if this contour was filtered out due to small area
            if i not in valid_indices:
                continue

            # --- Get contour and hierarchy for valid index ---
            # Access the original contour list using the valid index 'i'
            contour = all_contours_initial[i]
            current_hierarchy = all_hierarchy[i]
            parent_index = current_hierarchy[3]

            # --- Check if it's an inner contour AND its parent is also valid ---
            if parent_index != -1 and parent_index in valid_indices:
                loop_area = cv2.contourArea(contour) # Area already known to be >= MIN_CONTOUR_AREA_OVERALL

                # --- Apply specific LOOP Area Filter ---
                if loop_area >= LOOP_MIN_AREA:
                    # --- Check Parent Area (using original contour list for parent) ---
                    parent_contour = all_contours_initial[parent_index]
                    parent_area = cv2.contourArea(parent_contour) # Parent area already >= MIN_CONTOUR_AREA_OVERALL

                    if parent_area >= PARENT_MIN_AREA:
                        # --- Apply Corner/Vertex Count Filter ---
                        vertex_count = 0
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 1e-6:
                            try:
                                epsilon = APPROX_EPSILON_FACTOR * perimeter
                                approx_poly = cv2.approxPolyDP(contour, epsilon, True)
                                vertex_count = len(approx_poly)
                            except Exception: pass # Ignore errors

                        # Check vertex count
                        if 0 < vertex_count <= MAX_LOOP_VERTICES:
                            # --- Style Loop Detected ---
                            self.detected_loops.append(contour) # Store the actual contour
                            self.loop_areas.append(loop_area)
                            self.loop_vertex_counts.append(vertex_count)
                            parent_indices_with_loops.add(parent_index)

        # 4. Final calculations based on filtered results
        self.shapes_with_loops = len(parent_indices_with_loops)
        # Count only valid top-level shapes
        self.analyzed_top_level_shapes = sum(
            1 for idx, h in enumerate(all_hierarchy)
            if h[3] == -1 and idx in valid_indices # Must be top-level AND not filtered out
        )

    def _calculate_loop_statistics(self):
        """ Calculates summary statistics based on filtered loops. """
        total_loops_found = len(self.detected_loops)
        # shapes_with_loops and analyzed_top_level_shapes are now calculated in _find_contours_and_filter_loops

        # Avoid division by zero
        percentage_shapes = 0.0
        if self.analyzed_top_level_shapes > 0:
             percentage_shapes = (self.shapes_with_loops / self.analyzed_top_level_shapes) * 100

        avg_loops_per_shape = 0.0
        if self.shapes_with_loops > 0:
            avg_loops_per_shape = total_loops_found / self.shapes_with_loops

        metrics = {
            'total_loops_found': total_loops_found,
            'avg_loop_area': np.mean(self.loop_areas) if total_loops_found > 0 else 0,
            'avg_loop_vertex_count': np.mean(self.loop_vertex_counts) if total_loops_found > 0 else 0,
            'median_loop_vertex_count': np.median(self.loop_vertex_counts) if total_loops_found > 0 else 0,
            'shapes_with_loops': self.shapes_with_loops,
            'percentage_shapes_with_loops': percentage_shapes,
            'avg_loops_per_shape_with_loops': avg_loops_per_shape,
            'analyzed_top_level_shapes': self.analyzed_top_level_shapes,
        }
        return metrics

    def _generate_visualization(self, metrics):
        """ Generates visualization with the key metric bar chart. """
        graphs = []
        if self.img is None:
             print("Warning: Original color image not available for visualization.")
             return graphs # Cannot visualize without the color image

        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        vis_image = img_rgb.copy()

        # --- Fixed visualization parameters ---
        LOOP_COLOR_BGR = (0, 255, 255) # Cyan
        LOOP_THICKNESS = 2
        BLUR_KSIZE_DISPLAY = 3 # Matching the hardcoded blur size

        # Draw detected loops
        if self.detected_loops:
             cv2.drawContours(vis_image, self.detected_loops, -1, LOOP_COLOR_BGR, thickness=LOOP_THICKNESS)

        plt.figure("Loop Detection Analysis", figsize=(10, 8))

        # Subplot 1: Original
        plt.subplot(2, 2, 1); plt.imshow(img_rgb); plt.title("Original Image"); plt.axis('off')

        # Subplot 2: Preprocessed
        plt.subplot(2, 2, 2)
        if self.binary_image is not None:
            plt.imshow(self.binary_image, cmap='gray')
            plt.title(f"Preprocessed (Blur K={BLUR_KSIZE_DISPLAY}, Otsu)")
        else:
            plt.title("Preprocessed Image (Not Available)")
        plt.axis('off')

        # Subplot 3: Detected Loops
        plt.subplot(2, 2, 3); plt.imshow(vis_image)
        plt.title(f"Detected Style Loops ({metrics.get('total_loops_found', 0)}) Highlighted"); plt.axis('off')

        # Subplot 4: Key Metric Bar Chart
        plt.subplot(2, 2, 4)
        plt.title("Loop Style Prevalence")
        percentage = metrics.get('percentage_shapes_with_loops', 0.0)
        bars = plt.bar([''], [percentage], color='dodgerblue', width=0.4)
        plt.ylim(0, 105); plt.ylabel("Shapes with Style Loops (%)"); plt.xticks([])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f'{yval:.1f}%',
                     va='bottom', ha='center', fontsize=11, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        plt.tight_layout(pad=1.5)
        buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8'); plt.close()
        graphs.append(plot_base64)
        return graphs

    def analyze(self, debug=False):
        """
        Runs the analysis pipeline: preprocess, find loops, calculate stats.

        Parameters:
            debug (bool): If True, generates and includes visualization graphs.

        Returns:
            dict: Contains 'metrics' dictionary and 'graphs' list (if debug=True).
        """
        try:
            self._preprocess_image()
            self._find_contours_and_filter_loops()
            metrics = self._calculate_loop_statistics()
        except Exception as e:
            print(f"Error during analysis steps: {e}")
            # Return empty/default results on error
            return {'metrics': self._calculate_loop_statistics(), 'graphs': []} # Ensure metrics dict structure exists

        result = {'metrics': metrics, 'graphs': []}

        if debug:
            # Generate visualization only if metrics seem valid
            if metrics and metrics.get('analyzed_top_level_shapes') is not None :
                 try:
                    result['graphs'] = self._generate_visualization(metrics=metrics)
                 except Exception as e:
                    print(f"Error during visualization generation: {e}")
                    result['graphs'] = [] # Ensure graphs list exists even if viz fails
            else:
                print("Warning: Skipping visualization due to missing or invalid metrics.")
                result['graphs'] = []

        return result

# === Example usage ===
if __name__ == "__main__":
    # --- Image Selection ---
    image_path = r"C:\Users\Samson\Desktop\Coding\IPPR\NoteMercy_Extension\backend\atest\calligraphic.png"

    analyzer = LoopDetectionAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)

    print("\n===== Loop Detection Results =====")
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
