import base64
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte # Helper for skimage compatibility if needed

class StrokeTerminalAnalyzer:
    """
    Analyzes stroke terminals (endpoints) in an image of a single word
    using skeletonization and shape descriptors (Hu Moments) with fixed parameters.
    """

    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer by loading the original color image.
        """
        self.img_color = None
        if is_base64:
            try:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                self.img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if self.img_color is None:
                    raise ValueError("Could not decode base64 image as color.")
            except Exception as e:
                raise ValueError(f"Error decoding base64 image: {e}")
        else:
            self.img_color = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if self.img_color is None:
                raise ValueError(f"Could not read image as color at path: {image_input}")

        if self.img_color is None:
            raise ValueError("Image could not be loaded.")

        self.original_height, self.original_width = self.img_color.shape[:2]

        # Initialize results holders
        self._reset_analysis_data()

    def _reset_analysis_data(self):
        """
        Clears intermediate data and results from previous analysis runs.
        """
        self.binary_image = None
        self.skeleton = None
        self.endpoints = []  # List of (x, y) coordinates
        self.terminal_features = []  # List of feature dicts for each terminal
        self.metrics = {}  # Aggregate metrics

    def preprocess_image(self):
        """
        Applies preprocessing: Grayscale, Blur, Thresholding, Noise Reduction,
        and Morphological Closing using fixed internal parameters.
        Stores the binary result (white strokes=255, black background=0) in self.binary_image.
        """
        # --- Fixed Preprocessing Parameters ---
        _BLUR_KSIZE = 3          # Kernel size for Gaussian Blur (must be odd > 1, or <=1 to disable)
        _THRESH_VALUE = 127      # Threshold value for cv2.threshold
        _THRESH_MAX_VALUE = 255  # Max value for thresholding
        _THRESH_TYPE = cv2.THRESH_BINARY_INV # Invert: strokes become white
        _MORPH_CLOSE_KERNEL_SIZE = (5, 5) # Kernel size for morphological closing

        # 1. Grayscale
        gray_image = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        processed = gray_image.copy()

        # 2. Gaussian Blur (Optional)
        if _BLUR_KSIZE > 1:
            ksize = _BLUR_KSIZE if _BLUR_KSIZE % 2 != 0 else _BLUR_KSIZE + 1 # Ensure odd
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

        # 3. Thresholding
        ret, self.binary_image = cv2.threshold(processed, _THRESH_VALUE, _THRESH_MAX_VALUE, _THRESH_TYPE)

        # 4. Morphological Closing (to close small gaps)
        kernel = np.ones(_MORPH_CLOSE_KERNEL_SIZE, np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel)

    def _find_stroke_endpoints(self):
        """
        Performs skeletonization and identifies endpoints on the skeleton.
        Stores skeleton in self.skeleton and endpoints in self.endpoints.
        Requires scikit-image.
        """
        if self.binary_image is None:
            raise RuntimeError("Preprocessing must be run before finding endpoints.")

        # --- Fixed Endpoint Finding Parameters ---
        _SKELETON_PAD_VALUE = 0       # Value used for padding border (0=black)
        _ENDPOINT_NEIGHBOR_COUNT = 1  # An endpoint has exactly this many neighbors

        # Skeletonization requires white strokes on black background, bool or {0,1} type
        # Ensure image is binary 0 or 1 for skimage
        img_for_skeleton = self.binary_image > 128  # Convert to boolean (True/False)
        skeleton_bool = skeletonize(img_for_skeleton)
        self.skeleton = img_as_ubyte(skeleton_bool) # Convert back to uint8 {0, 255}

        # Find endpoints: Pixels with exactly one neighbor in 8-connectivity
        self.endpoints = []
        rows, cols = self.skeleton.shape

        # Pad the skeleton to avoid boundary checks inside the loop
        # Divide by 255 to work with {0, 1} values for easy summing
        skeleton_normalized = self.skeleton // 255
        skeleton_padded = cv2.copyMakeBorder(skeleton_normalized, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=_SKELETON_PAD_VALUE)

        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                # Check if the current pixel is part of the skeleton (value 1)
                if skeleton_padded[r, c] == 1:
                    # Sum the 3x3 neighborhood
                    neighbors_sum = np.sum(skeleton_padded[r - 1:r + 2, c - 1:c + 2])
                    # Subtract the center pixel itself (which is 1) to get the count of neighbors
                    num_neighbors = neighbors_sum - 1

                    # Check if it's an endpoint
                    if num_neighbors == _ENDPOINT_NEIGHBOR_COUNT:
                        # Adjust coordinates back to original image space (due to padding)
                        self.endpoints.append((c - 1, r - 1))  # Store as (x, y)

    def _analyze_terminals(self):
        """
        Analyzes the region around each detected endpoint in the *binary* image
        using fixed ROI size. Calculates Hu Moments for each terminal ROI.
        Stores individual terminal features and calculates aggregate metrics.
        """
        # --- Fixed Terminal Analysis Parameters ---
        _ROI_SIZE = 15         # Side length of the square ROI (should be odd)
        _LOG_HU_EPSILON = 1e-7 # Small value to avoid log(0) or log(negative)

        if not self.endpoints:
            print("Warning: No endpoints found to analyze.")
            self.metrics = {'terminal_count': 0}
            # Initialize stats to 0 if no terminals
            for i in range(7):
                self.metrics[f'mean_log_hu_{i + 1}'] = 0.0
                self.metrics[f'std_dev_log_hu_{i + 1}'] = 0.0
            return

        # Ensure ROI size is odd
        roi_size = _ROI_SIZE if _ROI_SIZE % 2 != 0 else _ROI_SIZE + 1
        if roi_size != _ROI_SIZE:
             print(f"Warning: ROI_SIZE should be odd. Adjusting from {_ROI_SIZE} to {roi_size}.")

        half_roi = roi_size // 2
        self.terminal_features = []
        all_hu_moments = []  # Collect log-transformed Hu moments for stats

        if self.binary_image is None:
            raise RuntimeError("Binary image is not available for terminal analysis.")

        for (x, y) in self.endpoints:
            # Define ROI boundaries, clamping to image dimensions
            r_start = max(0, y - half_roi)
            r_end = min(self.original_height, y + half_roi + 1)
            c_start = max(0, x - half_roi)
            c_end = min(self.original_width, x + half_roi + 1)

            # Extract ROI from the original *binary* image
            roi = self.binary_image[r_start:r_end, c_start:c_end]

            if roi.size == 0 or np.sum(roi) == 0:  # Skip empty or fully black ROIs
                print(f"Warning: Skipping empty ROI at endpoint ({x},{y})")
                continue

            # Calculate moments and Hu Moments for the ROI
            moments = cv2.moments(roi)
            hu_moments = cv2.HuMoments(moments).flatten()

            # Log-transform Hu moments
            log_hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + _LOG_HU_EPSILON)

            terminal_data = {
                'coords': (x, y),
                'roi_shape': roi.shape,
                'hu_moments': hu_moments.tolist(),
                'log_hu_moments': log_hu_moments.tolist()
            }
            self.terminal_features.append(terminal_data)
            all_hu_moments.append(log_hu_moments)

        # Calculate Aggregate Metrics
        self.metrics['terminal_count'] = len(self.terminal_features)

        if self.terminal_features:
            all_hu_moments_np = np.array(all_hu_moments)
            mean_hu = np.mean(all_hu_moments_np, axis=0)
            std_dev_hu = np.std(all_hu_moments_np, axis=0)

            for i in range(7):
                self.metrics[f'mean_log_hu_{i + 1}'] = mean_hu[i]
                self.metrics[f'std_dev_log_hu_{i + 1}'] = std_dev_hu[i]
        else: # Handle case where endpoints were found but all ROIs were invalid
            for i in range(7):
                self.metrics[f'mean_log_hu_{i + 1}'] = 0.0
                self.metrics[f'std_dev_log_hu_{i + 1}'] = 0.0

    def _generate_visualization(self):
        """
        Generates visualization plots showing detected terminals using fixed settings.
        Returns a list containing a single base64 encoded PNG string.
        Requires matplotlib.
        """
        # --- Fixed Visualization Parameters ---
        _FIGURE_SIZE = (12, 10)
        _ENDPOINT_MARKER_COLOR_BGR = (0, 0, 255)  # Red in BGR for OpenCV drawing
        _ENDPOINT_MARKER_RADIUS = 10               # Radius for cv2.circle
        _ENDPOINT_MARKER_THICKNESS = 2            # Thickness for cv2.circle
        _LAYOUT_PADDING = 1.5

        graphs = []
        if self.img_color is None or self.binary_image is None or self.skeleton is None:
            print("Warning: Cannot generate visualization, required image data missing.")
            return graphs

        try:
            plt.figure("Stroke Terminal Analysis", figsize=_FIGURE_SIZE)

            # Plot 1: Original Image
            plt.subplot(2, 2, 1)
            img_rgb = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title("Original Image")
            plt.axis('off')

            # Plot 2: Binary Image
            plt.subplot(2, 2, 2)
            plt.imshow(self.binary_image, cmap='gray')
            plt.title("Binary Image (Strokes in White)")
            plt.axis('off')

            # Plot 3: Skeleton
            plt.subplot(2, 2, 3)
            plt.imshow(self.skeleton, cmap='gray')
            plt.title("Skeleton")
            plt.axis('off')

            # Plot 4: Terminals Marked on Original Image
            plt.subplot(2, 2, 4)
            vis_img_terminals = self.img_color.copy() # Work on a copy
            for (x, y) in self.endpoints:
                cv2.circle(vis_img_terminals, (x, y), _ENDPOINT_MARKER_RADIUS,
                           _ENDPOINT_MARKER_COLOR_BGR, _ENDPOINT_MARKER_THICKNESS)

            vis_img_terminals_rgb = cv2.cvtColor(vis_img_terminals, cv2.COLOR_BGR2RGB)
            plt.imshow(vis_img_terminals_rgb)
            plt.title(f"Detected Terminals ({len(self.endpoints)}) Marked")
            plt.axis('off')

            plt.tight_layout(pad=_LAYOUT_PADDING)

            # Save plot to buffer and encode
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close() # Close the figure explicitly

            graphs.append(plot_base64)

        except Exception as e:
            print(f"Error during visualization generation: {e}")
            plt.close() # Ensure figure is closed on error
        finally:
             # Ensure plot is closed if it exists, even without explicit error handling block
            if plt.fignum_exists("Stroke Terminal Analysis"):
                plt.close("Stroke Terminal Analysis")

        return graphs

    def analyze(self, debug=False):
        """
        Orchestrates the analysis process using fixed internal parameters:
        preprocess, find endpoints, analyze terminals.

        Args:
            debug (bool): If True, generate and include visualization graphs in the result.

        Returns:
            dict: A dictionary containing 'metrics', 'preprocessed_image' (base64),
                  and 'graphs' (list of base64, only if debug=True).
                  The 'metrics' dict includes terminal count and statistics on
                  log-transformed Hu moments. It may also contain an 'error' key.
        """
        self._reset_analysis_data()  # Clear previous results
        result = {'metrics': {}, 'preprocessed_image': None, 'graphs': []}

        try:
            # Call methods without passing parameters - they use internal constants now
            self.preprocess_image()
            self._find_stroke_endpoints()
            self._analyze_terminals()

            # Copy metrics to result *after* all steps succeed
            result['metrics'] = self.metrics.copy()

        except Exception as e:
            print(f"Error during analysis pipeline: {e}")
            # Store error message in metrics
            result['metrics']['error'] = str(e)
            result['metrics']['terminal_count'] = 0 # Ensure count is 0 on error
            # Initialize stats to 0 on error if not already set
            if 'mean_log_hu_1' not in result['metrics']:
                for i in range(7):
                    result['metrics'][f'mean_log_hu_{i + 1}'] = 0.0
                    result['metrics'][f'std_dev_log_hu_{i + 1}'] = 0.0
            # Return early if a critical error occurred
            return result

        # Generate visualization only if debug is True AND no critical error occurred
        if debug:
            try:
                result['graphs'] = self._generate_visualization()
            except Exception as e:
                print(f"Error generating visualization after successful analysis: {e}")
                # Optionally add a note about viz error to metrics
                result['metrics']['visualization_error'] = str(e)

        # Add preprocessed image to result (regardless of debug, useful for inspection)
        if self.binary_image is not None:
            try:
                is_success, buffer = cv2.imencode('.png', self.binary_image)
                if is_success:
                    result['preprocessed_image'] = base64.b64encode(buffer).decode('utf-8')
                else:
                     print("Warning: Failed to encode preprocessed image.")
            except Exception as e:
                print(f"Error encoding preprocessed image: {e}")


        return result

# === Example usage (Updated) ===
if __name__ == "__main__":
    # --- Image Selection ---
    image_path = r"C:\Users\Samson\Desktop\Coding\IPPR\NoteMercy_Extension\backend\atest\cursive2.png"
    analyzer = StrokeTerminalAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)

    print("\n===== Stroke Terminal Analysis Results =====")
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
