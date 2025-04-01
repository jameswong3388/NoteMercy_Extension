import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class SymbolDensityAnalyzer:
    """
    Analyzes the overall stroke density within a resized image.
    Resizes all input images to a fixed dimension (900x383) and calculates
    the density of stroke pixels (white) relative to the total area of the
    resized image. Useful for comparing overall 'ink coverage' across images
    normalized to the same size.
    """
    FIXED_WIDTH = 900
    FIXED_HEIGHT = 383

    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer by loading the original color image and
        resizing it to the fixed dimensions (900x383).
        """
        self.img_color_original = None  # Store original before resize just in case
        self.img_color = None  # This will hold the resized image
        self.image_width = self.FIXED_WIDTH
        self.image_height = self.FIXED_HEIGHT

        temp_img = None
        try:
            if is_base64:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                temp_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if temp_img is None:
                    raise ValueError("Could not decode base64 image as color.")
            else:
                temp_img = cv2.imread(image_input, cv2.IMREAD_COLOR)
                if temp_img is None:
                    raise ValueError(f"Could not read image as color at path: {image_input}")

            self.img_color_original = temp_img.copy()

            # --- Resize the image ---
            self.img_color = cv2.resize(temp_img, (self.FIXED_WIDTH, self.FIXED_HEIGHT), interpolation=cv2.INTER_AREA)

            self._reset_analysis_data()

        except Exception as e:
            raise ValueError(f"Error loading or resizing image: {e}")

    def _reset_analysis_data(self):
        """
        Clears intermediate data from previous analysis runs.
        """
        self.binary_image = None
        self.contours = []  # Still find contours for potential visualization
        self.analysis_metrics = {}
        self.plot_data = {'contours': []}  # Simplified plot data

    def preprocess_image(self):
        """
        Applies preprocessing: Grayscale, Blur, Thresholding, Noise Reduction,
        and Morphological Closing using fixed internal parameters.
        Stores the binary result (white strokes=255, black background=0) in self.binary_image.
        """
        # --- Fixed Preprocessing Parameters ---
        _BLUR_KSIZE = 3  # Kernel size for Gaussian Blur (must be odd > 1, or <=1 to disable)
        _THRESH_VALUE = 127  # Threshold value for cv2.threshold
        _THRESH_MAX_VALUE = 255  # Max value for thresholding
        _THRESH_TYPE = cv2.THRESH_BINARY_INV  # Invert: strokes become white

        # 1. Grayscale
        gray_image = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        processed = gray_image.copy()

        # 2. Gaussian Blur (Optional)
        if _BLUR_KSIZE > 1:
            ksize = _BLUR_KSIZE if _BLUR_KSIZE % 2 != 0 else _BLUR_KSIZE + 1  # Ensure odd
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

        # 3. Thresholding
        ret, self.binary_image = cv2.threshold(processed, _THRESH_VALUE, _THRESH_MAX_VALUE, _THRESH_TYPE)

    def calculate_density(self):
        """
        Calculates the overall stroke density for the entire resized image.
        Density = (Total number of stroke pixels) / (Total fixed image area).
        Optionally finds contours for visualization purposes.

        Returns:
            tuple: (metrics_dict, plot_data_dict)
                   metrics_dict contains the overall density and pixel counts.
                   plot_data_dict contains contours found (if any).
        """
        metrics = {
            'total_image_pixels': self.image_width * self.image_height,
            'total_stroke_pixels': 0,
            'symbol_density': 0.0,
            'total_contours_found': 0  # Keep track of contours found
        }
        plot_data = {'contours': []}  # Reset plot data

        if self.binary_image is None:
            metrics['error'] = "Binary image not available for density calculation."
            return metrics, plot_data

        # Check if binary image dimensions match fixed size (sanity check)
        h, w = self.binary_image.shape
        if h != self.image_height or w != self.image_width:
            metrics[
                'error'] = f"Binary image dimensions ({w}x{h}) do not match fixed size ({self.image_width}x{self.image_height})."
            # Optionally try to proceed anyway or return error
            # Let's proceed but log a warning potentially
            print(
                f"Warning: Binary image dimensions ({w}x{h}) differ from target ({self.image_width}x{self.image_height}).")
            # Update total pixels based on actual binary image size if proceeding
            metrics['total_image_pixels'] = w * h

        # Calculate total white pixels (strokes) in the entire binary image
        total_stroke_pixels = cv2.countNonZero(self.binary_image)
        metrics['total_stroke_pixels'] = total_stroke_pixels

        # Calculate overall density
        total_area = metrics['total_image_pixels']
        if total_area > 0:
            metrics['symbol_density'] = total_stroke_pixels / total_area
        else:
            metrics['symbol_density'] = 0.0

        # Find contours - primarily for visualization now, not density calculation
        self.contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        metrics['total_contours_found'] = len(self.contours)
        plot_data['contours'] = self.contours  # Store raw contours

        self.analysis_metrics = metrics  # Store metrics
        self.plot_data = plot_data  # Store plot data
        return metrics, plot_data

    def _generate_visualization(self, metrics, plot_data):
        """
        Generates visualization plots showing the resized image, binary image,
        and contours found. Highlights the overall stroke density.
        Uses fixed display parameters.

        Args:
            metrics (dict): Dictionary of calculated metrics.
            plot_data (dict): Dictionary containing contours for plotting.

        Returns:
            list: List containing base64 encoded string of the visualization plot.
        """
        # --- Fixed Visualization Parameters ---
        FIGURE_SIZE = (12, 6)  # Adjusted figure size
        CONTOUR_COLOR = (0, 255, 0)  # Green in BGR
        CONTOUR_THICKNESS = 1
        LAYOUT_PADDING = 1.5

        graphs = []
        if self.img_color is None or self.binary_image is None:
            print("Warning: Cannot generate visualization, required image data missing.")
            return graphs

        plt.figure(f"Overall Symbol Density Analysis ({self.image_width}x{self.image_height})", figsize=FIGURE_SIZE)

        # Plot 1: Resized Original Image
        plt.subplot(1, 3, 1)
        img_rgb = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"Resized Image ({self.image_width}x{self.image_height})")
        plt.axis('off')

        # Plot 2: Preprocessed Image (Binary)
        plt.subplot(1, 3, 2)
        plt.imshow(self.binary_image, cmap='gray')
        plt.title("Preprocessed (Strokes White)")
        plt.axis('off')

        # Plot 3: Contours on Resized Image
        plt.subplot(1, 3, 3)
        vis_img_contours = img_rgb.copy()
        contours = plot_data.get('contours', [])
        if contours:
            # Draw all found contours
            cv2.drawContours(vis_img_contours, contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

        plt.imshow(vis_img_contours)
        density = metrics.get('symbol_density', 0.0)
        num_contours = metrics.get('total_contours_found', 0)
        plt.title(f"Contours ({num_contours}) / Overall Density: {density:.4f}")
        plt.axis('off')

        plt.tight_layout(pad=LAYOUT_PADDING)

        # Save plot to buffer and encode
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()  # Close the figure explicitly

        graphs.append(plot_base64)
        return graphs

    def analyze(self, debug=False):
        """
        Orchestrates the overall stroke density analysis process on the resized image.

        Args:
            debug (bool): If True, generates and includes visualization graphs.

        Returns:
            dict: A dictionary containing:
                  'metrics': The calculated overall density metrics.
                  'preprocessed_image': Base64 encoded string of the binary image.
                  'graphs': List containing base64 encoded visualization(s) if debug=True.
        """
        try:
            self._reset_analysis_data()  # Resets data, image already resized in init
            self.preprocess_image()  # Preprocess the resized image
            metrics, plot_data = self.calculate_density()  # Calculate density on resized binary img

            result = {'metrics': metrics, 'graphs': []}

            if 'error' in metrics:
                print(f"Error during analysis: {metrics['error']}")
                # Still return preprocessed image if available
                if self.binary_image is not None:
                    _, buffer = cv2.imencode('.png', self.binary_image)
                    preprocessed_image_base64 = base64.b64encode(buffer).decode('utf-8')
                    result['preprocessed_image'] = preprocessed_image_base64
                return result

            if debug:
                # Generate visualization using the resized image data and found contours
                result['graphs'] = self._generate_visualization(metrics, plot_data)

            # Include the preprocessed image (binary, fixed size) in the results
            if self.binary_image is not None:
                _, buffer = cv2.imencode('.png', self.binary_image)
                preprocessed_image_base64 = base64.b64encode(buffer).decode('utf-8')
                result['preprocessed_image'] = preprocessed_image_base64
            else:
                result['preprocessed_image'] = None

            return result

        except Exception as e:
            # Catch potential errors during the process
            print(f"An error occurred during analysis: {e}")
            return {'metrics': {'error': str(e)}, 'graphs': [], 'preprocessed_image': None}


# === Example Usage ===
if __name__ == "__main__":
    # Use a sample image path - CHANGE THIS TO YOUR IMAGE
    image_path = r"C:\Users\Samson\Desktop\Coding\IPPR\NoteMercy_Extension\backend\atest\shorthand1.png"
    analyzer = SymbolDensityAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)

    print("\n===== Symbol Density Analysis Results =====")
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
