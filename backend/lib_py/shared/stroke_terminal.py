import base64
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize


class StrokeTerminalAnalyzer:
    """
    Analyzes stroke terminals (endpoints) in an image of a single word
    using skeletonization and shape descriptors (Hu Moments) without OCR
    or neural networks.

    Assumes input is an image (path or base64) containing a single word
    with strokes reasonably contrasted against the background.
    """

    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer by loading the original color image.

        Args:
            image_input (str): Either a file path to the image or a base64 encoded string.
            is_base64 (bool): True if image_input is base64, False if it's a file path.
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

    def preprocess_image(self, blur_ksize=3, threshold_block_size=11, threshold_c=2, noise_reduction_kernel=3):
        """
        Applies preprocessing: Grayscale, Blur, Adaptive Thresholding, and noise reduction.
        Stores the binary result (white strokes=255, black background=0) in self.binary_image.

        Args:
            blur_ksize (int): Kernel size for Gaussian Blur (odd number > 1, or 0/1 to disable).
            threshold_block_size (int): Size of the neighborhood area for adaptive thresholding (odd number > 1).
            threshold_c (int): Constant subtracted from the mean in adaptive thresholding.
            noise_reduction_kernel (int): Kernel size for noise reduction (odd number > 1, or 0/1 to disable).
        """
        if self.img_color is None:
            raise RuntimeError("Cannot preprocess, image not loaded correctly.")

        # 1. Grayscale
        gray_image = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        processed = gray_image.copy()

        # 2. Gaussian Blur (Optional)
        if blur_ksize > 1:
            # Ensure kernel size is odd
            ksize = blur_ksize if blur_ksize % 2 != 0 else blur_ksize + 1
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

        # 3. Adaptive Thresholding (Inverse Binary)
        # Ensures strokes are white (255) on black (0) background
        # Check block size validity
        if threshold_block_size <= 1 or threshold_block_size % 2 == 0:
            print(
                f"Warning: threshold_block_size ({threshold_block_size}) is invalid. Needs to be odd and > 1. Using 11.")
            threshold_block_size = 11

        self.binary_image = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, threshold_block_size, threshold_c
        )

        # 4. Noise Reduction (Optional)
        if noise_reduction_kernel > 1:
            ksize = noise_reduction_kernel if noise_reduction_kernel % 2 != 0 else noise_reduction_kernel + 1
            self.binary_image = cv2.medianBlur(self.binary_image, ksize)

        # Optional: Morphological Opening to remove small noise specks
        # You might adjust the kernel size depending on the noise level
        kernel = np.ones((5, 5), np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)

    def _find_stroke_endpoints(self):
        """
        Performs skeletonization and identifies endpoints on the skeleton.
        Stores skeleton in self.skeleton and endpoints in self.endpoints.
        Requires scikit-image.
        """
        if self.binary_image is None:
            raise RuntimeError("Preprocessing must be run before finding endpoints.")

        # Skeletonization requires white strokes on black background, bool or {0,1} type
        # Ensure image is binary 0 or 1 for skimage
        img_for_skeleton = self.binary_image > 128  # Convert to boolean (True/False)
        skeleton_bool = skeletonize(img_for_skeleton)
        self.skeleton = skeleton_bool.astype(np.uint8) * 255  # Convert back to uint8 {0, 255}

        # Find endpoints: Pixels with exactly one neighbor in 8-connectivity
        self.endpoints = []
        rows, cols = self.skeleton.shape
        # Pad the skeleton to avoid boundary checks inside the loop
        # Use // 255 division for finding neighbors in the {0, 255} image
        skeleton_padded = cv2.copyMakeBorder(self.skeleton // 255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                # Check if the current pixel is part of the skeleton (value 1 after division)
                if skeleton_padded[r, c] == 1:
                    # Count neighbors in the 3x3 window around the pixel
                    neighbors_sum = np.sum(skeleton_padded[r - 1:r + 2, c - 1:c + 2])
                    # Subtract the center pixel itself to get the count of neighbors
                    num_neighbors = neighbors_sum - 1

                    if num_neighbors == 1:
                        # Adjust coordinates back to original image space (due to padding)
                        self.endpoints.append((c - 1, r - 1))  # (x, y) format

    def _analyze_terminals(self, roi_size=15):
        """
        Analyzes the region around each detected endpoint in the *binary* image.
        Calculates Hu Moments for each terminal ROI.
        Stores individual terminal features and calculates aggregate metrics.

        Args:
            roi_size (int): The side length of the square ROI around each endpoint. Should be odd and > 0.
        """
        if not self.endpoints:
            print("Warning: No endpoints found to analyze.")
            self.metrics = {'terminal_count': 0}
            # Initialize stats to 0 if no terminals
            for i in range(7):
                self.metrics[f'mean_log_hu_{i + 1}'] = 0.0
                self.metrics[f'std_dev_log_hu_{i + 1}'] = 0.0
            return

        if roi_size <= 0:
            print("Warning: roi_size must be > 0. Using default of 15.")
            roi_size = 15
        if roi_size % 2 == 0:
            roi_size += 1  # Ensure odd size for centering
            print(f"Warning: roi_size should be odd. Adjusting to {roi_size}.")

        half_roi = roi_size // 2
        self.terminal_features = []
        all_hu_moments = []  # Collect log-transformed Hu moments for statistical analysis

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

            # Log-transform Hu moments for better scale invariance and stability
            # Add small epsilon to avoid log(0) or log(negative) if moments are weird
            log_hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)

            terminal_data = {
                'coords': (x, y),
                'roi_shape': roi.shape,  # Store ROI shape for debug if needed
                'hu_moments': hu_moments.tolist(),
                'log_hu_moments': log_hu_moments.tolist()
                # Avoid storing 'roi' itself unless needed for deep debugging, can consume memory
            }
            self.terminal_features.append(terminal_data)
            all_hu_moments.append(log_hu_moments)  # Use log-transformed for stats

        # Calculate Aggregate Metrics
        self.metrics['terminal_count'] = len(self.terminal_features)

        if self.terminal_features:
            all_hu_moments_np = np.array(all_hu_moments)
            # Calculate mean and std dev for each of the 7 Hu moments
            mean_hu = np.mean(all_hu_moments_np, axis=0)
            std_dev_hu = np.std(all_hu_moments_np, axis=0)

            for i in range(7):
                self.metrics[f'mean_log_hu_{i + 1}'] = mean_hu[i]
                self.metrics[f'std_dev_log_hu_{i + 1}'] = std_dev_hu[i]
        else:  # Handle case where endpoints were found but all ROIs were empty
            for i in range(7):
                self.metrics[f'mean_log_hu_{i + 1}'] = 0.0
                self.metrics[f'std_dev_log_hu_{i + 1}'] = 0.0

    def _generate_visualization(self):
        """
        Generates visualization plots showing detected terminals.
        Returns a list containing a single base64 encoded PNG string.
        Requires matplotlib.
        """
        # --- Fixed Visualization Parameters ---
        FIGURE_SIZE = (12, 10)
        ENDPOINT_MARKER_COLOR_BGR = (0, 0, 255)  # Red in BGR for OpenCV drawing
        ENDPOINT_MARKER_RADIUS = 5  # Radius for cv2.circle
        ENDPOINT_MARKER_THICKNESS = 1  # Thickness for cv2.circle
        LAYOUT_PADDING = 1.5

        graphs = []
        if self.img_color is None or self.binary_image is None or self.skeleton is None:
            print("Warning: Cannot generate visualization, required image data missing.")
            return graphs

        try:
            plt.figure("Stroke Terminal Analysis", figsize=FIGURE_SIZE)

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
            vis_img_terminals = self.img_color.copy()  # Work on a copy
            for (x, y) in self.endpoints:
                # Use OpenCV drawing for consistency
                cv2.circle(vis_img_terminals, (x, y), ENDPOINT_MARKER_RADIUS, ENDPOINT_MARKER_COLOR_BGR,
                           ENDPOINT_MARKER_THICKNESS)

            vis_img_terminals_rgb = cv2.cvtColor(vis_img_terminals, cv2.COLOR_BGR2RGB)
            plt.imshow(vis_img_terminals_rgb)
            plt.title(f"Detected Terminals ({len(self.endpoints)}) Marked")
            plt.axis('off')

            plt.tight_layout(pad=LAYOUT_PADDING)

            # Save plot to buffer and encode
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.show()
            plt.close()  # Close the figure explicitly to free memory

            graphs.append(plot_base64)

        except Exception as e:
            print(f"Error during visualization generation: {e}")
            plt.close()  # Ensure figure is closed even if error occurs
            # Optionally return an empty list or re-raise

        return graphs

    def analyze(self, debug=False, **kwargs):
        """
        Orchestrates the analysis process: preprocess, find endpoints, analyze terminals.

        Args:
            debug (bool): If True, generate and include visualization graphs.
            **kwargs: Parameters to pass to internal methods like preprocess_image
                      (e.g., blur_ksize, threshold_block_size, threshold_c, noise_reduction_kernel)
                      and _analyze_terminals (e.g., roi_size).

        Returns:
            dict: A dictionary containing 'metrics' and 'graphs' (if debug=True).
                  The 'metrics' dict includes terminal count and statistics on
                  log-transformed Hu moments. It may also contain an 'error' key.
        """
        self._reset_analysis_data()  # Clear previous results
        result = {'metrics': {}, 'graphs': []}

        try:
            # Pass preprocessing parameters if provided
            preproc_args = {k: v for k, v in kwargs.items() if
                            k in ['blur_ksize', 'threshold_block_size', 'threshold_c', 'noise_reduction_kernel']}
            self.preprocess_image(**preproc_args)

            self._find_stroke_endpoints()

            # Pass analysis parameters if provided
            analysis_args = {k: v for k, v in kwargs.items() if k in ['roi_size']}
            self._analyze_terminals(**analysis_args)

            # Copy metrics to result *after* all steps succeed
            result['metrics'] = self.metrics.copy()

        except Exception as e:
            print(f"Error during analysis pipeline: {e}")
            # Store error message in metrics
            result['metrics']['error'] = str(e)
            result['metrics']['terminal_count'] = 0  # Ensure count is 0 on error
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

        # Add preprocessed image to result
        if self.binary_image is not None:
            _, buffer = cv2.imencode('.png', self.binary_image)
            result['preprocessed_image'] = base64.b64encode(buffer).decode('utf-8')
        else:
            result['preprocessed_image'] = None

        return result

# === Example Usage ===
if __name__ == "__main__":
    image_path = "../../atest/print3.png"

    analyzer = StrokeTerminalAnalyzer(image_path, is_base64=False)

    # Example: Run analysis with custom parameters and debug visualization enabled
    # Adjust these parameters based on your image characteristics (e.g., shorthand)
    analysis_results = analyzer.analyze(
        debug=True,  # Generate graphs
        blur_ksize=0,  # Less blur might be better for sharp details
        threshold_block_size=15,  # Adjust based on stroke thickness/image size
        threshold_c=5,  # Adjust based on contrast
        roi_size=21,  # Adjust ROI size to capture terminal features
        noise_reduction_kernel=3 # add noise reduction
    )

    print(analysis_results['metrics'])