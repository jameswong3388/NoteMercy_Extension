import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class StrokeWidthAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer by loading the original color image.
        Handles both base64 string and file path inputs.
        """
        self.img_color = None
        self.gray_img = None
        self.binary_image = None
        self.distance_transform = None
        self.skeleton = None
        self.stroke_radii = np.array([])  # Initialize as empty array

        try:
            if is_base64:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                # Load as color image initially
                self.img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if self.img_color is None:
                    # Attempt to decode as grayscale if color failed (e.g., single channel input)
                    self.img_color = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if self.img_color is not None:
                        # If successful, convert grayscale back to BGR for consistency
                        self.img_color = cv2.cvtColor(self.img_color, cv2.COLOR_GRAY2BGR)
                    else:
                        raise ValueError("Could not decode base64 image as color or grayscale.")
            else:
                # Load as color image initially
                self.img_color = cv2.imread(image_input, cv2.IMREAD_COLOR)
                if self.img_color is None:
                    # Attempt to load as grayscale if color failed
                    self.img_color = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
                    if self.img_color is not None:
                        # Convert grayscale back to BGR
                        self.img_color = cv2.cvtColor(self.img_color, cv2.COLOR_GRAY2BGR)
                    else:
                        raise ValueError(f"Could not read image as color or grayscale at path: {image_input}")

            # Basic check for valid image dimensions
            if len(self.img_color.shape) < 2 or self.img_color.shape[0] == 0 or self.img_color.shape[1] == 0:
                raise ValueError("Loaded image has invalid dimensions.")

            self.original_height, self.original_width = self.img_color.shape[:2]

        except Exception as e:
            raise ValueError(f"Error during image loading: {e}")

        self._reset_analysis_data()  # Initialize other instance vars

    def _reset_analysis_data(self):
        """
        Clears intermediate data from previous analysis runs.
        """
        # Keep self.img_color, but reset processed data
        self.gray_img = None
        self.binary_image = None
        self.distance_transform = None
        self.skeleton = None
        self.stroke_radii = np.array([])

    def preprocess_image(self):
        """
        Applies preprocessing: Grayscale conversion, Gaussian Blur, Otsu Thresholding.
        Ensures text is white on black background.
        Stores results in self.gray_img and self.binary_image (boolean).
        Uses fixed internal parameters.
        """
        # --- Fixed Preprocessing Parameters ---
        BLUR_KSIZE = 5  # Kernel size for Gaussian Blur (odd number)

        if self.img_color is None:
            raise RuntimeError("Original color image is not available for preprocessing.")

        # 1. Convert to Grayscale
        # Check if it's already effectively grayscale (3 channels but all same)
        if len(self.img_color.shape) == 3 and self.img_color.shape[2] == 3:
            # Efficient check for grayscale stored in BGR format
            b, g, r = cv2.split(self.img_color)
            if np.array_equal(b, g) and np.array_equal(g, r):
                self.gray_img = b  # Use one channel
            else:
                self.gray_img = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        elif len(self.img_color.shape) == 2:  # Input was already grayscale
            self.gray_img = self.img_color.copy()  # Should not happen based on __init__ logic, but safe check
        else:
            raise ValueError("Cannot convert image to grayscale - unexpected format.")

        # 2. Gaussian Blur (Applied to the grayscale image)
        processed = cv2.GaussianBlur(self.gray_img, (BLUR_KSIZE, BLUR_KSIZE), 0)

        # 3. Otsu's Thresholding
        thresh_val = threshold_otsu(processed)
        # Invert threshold to get white text on black background
        _, bw = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # 4. Store as boolean array (True for foreground/text)
        self.binary_image = bw > 0

    def _calculate_stroke_widths(self):
        """
        Calculates stroke widths using distance transform and skeletonization.
        Assumes self.binary_image is already computed.
        Stores intermediate results (distance transform, skeleton) and final radii.
        Returns the calculated metrics dictionary.
        """
        metrics = {
            'mean_width': 0.0,
            'width_std': 0.0,
            'width_ratio': 0.0,
            'variation_coefficient': 0.0
        }

        if self.binary_image is None or not np.any(self.binary_image):
            print("Warning: Binary image is empty or not available for stroke width calculation.")
            return metrics  # Return default zero metrics

        try:
            # 1. Compute Distance Transform (distance from background)
            self.distance_transform = distance_transform_edt(self.binary_image)

            # 2. Compute Skeleton
            self.skeleton = skeletonize(self.binary_image)

            # 3. Get Stroke Radii at Skeleton Points
            radii = self.distance_transform[self.skeleton]

            # Flatten and filter out zeros and non-finite values
            self.stroke_radii = radii.flatten()
            self.stroke_radii = self.stroke_radii[(self.stroke_radii > 0) & np.isfinite(self.stroke_radii)]

            # 4. Calculate Metrics if radii were found
            if self.stroke_radii.size > 0:
                mean_radius = np.mean(self.stroke_radii)
                std_radius = np.std(self.stroke_radii)
                min_radius = np.min(self.stroke_radii)
                max_radius = np.max(self.stroke_radii)

                metrics['mean_width'] = 2 * mean_radius
                metrics['width_std'] = 2 * std_radius
                metrics['width_ratio'] = max_radius / min_radius if min_radius > 0 else 0.0
                metrics['variation_coefficient'] = metrics['width_std'] / metrics['mean_width'] if metrics[
                                                                                                       'mean_width'] > 0 else 0.0

        except Exception as e:
            print(f"Error during stroke width calculation: {e}")
            metrics = {k: 0.0 for k in metrics}
            self.distance_transform = None
            self.skeleton = None
            self.stroke_radii = np.array([])

        return metrics

    def _generate_visualization(self, metrics):
        """
        Generates visualization plots using instance data and calculated metrics.
        Uses fixed display parameters.
        """
        # --- Fixed Visualization Parameters ---
        FIGURE_SIZE = (12, 10)
        SKELETON_COLOR = 'r'
        SKELETON_POINT_SIZE = 1
        HIST_BINS = 20
        MEAN_LINE_COLOR = 'r'
        LAYOUT_PADDING = 1.0

        graphs = []
        if self.img_color is None or self.binary_image is None or self.skeleton is None:
            print("Warning: Cannot generate visualization, required image data missing.")
            return graphs

        plt.figure("Stroke Width Analysis", figsize=FIGURE_SIZE)
        plt.suptitle('Stroke Width Analysis', fontsize=16)

        # Plot 1: Original Image
        plt.subplot(2, 2, 1)
        img_rgb = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title('Original Image')
        plt.axis('off')

        # Plot 2: Binary Image
        plt.subplot(2, 2, 2)
        plt.imshow(self.binary_image, cmap='gray')
        plt.title('Preprocessed Image')
        plt.axis('off')

        # Plot 3: Skeleton Points
        plt.subplot(2, 2, 3)
        plt.imshow(self.binary_image, cmap='gray')  # Show skeleton on binary background
        y_coords, x_coords = np.nonzero(self.skeleton)
        plt.scatter(x_coords, y_coords, s=SKELETON_POINT_SIZE, c=SKELETON_COLOR)
        plt.title('Skeleton Points')
        plt.axis('off')
        plt.xlim(0, self.original_width)
        plt.ylim(self.original_height, 0)

        # Plot 4: Stroke Width Distribution
        plt.subplot(2, 2, 4)
        if self.stroke_radii.size > 0:
            widths = 2 * self.stroke_radii
            plt.hist(widths, bins=HIST_BINS, edgecolor='black', alpha=0.7)
            mean_w = metrics.get('mean_width', 0.0)
            ratio = metrics.get('width_ratio', 0.0)
            plt.axvline(mean_w, color=MEAN_LINE_COLOR, linestyle='--', linewidth=2,
                        label=f'Mean: {mean_w:.2f}')
            plt.title(f'Stroke Width Distribution\nMean: {mean_w:.2f}, Ratio: {ratio:.2f}')
            plt.legend(fontsize='small')
        else:
            plt.hist([], bins=HIST_BINS, range=(0, 1))
            plt.title('Stroke Width Distribution\nNo valid stroke points found.')

        plt.xlabel('Stroke Width')
        plt.ylabel('Frequency')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=LAYOUT_PADDING)

        try:
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            graphs.append(plot_base64)
        except Exception as e:
            print(f"Error saving plot to buffer: {e}")
        finally:
            plt.close()

        return graphs

    def analyze(self, debug=False):
        """
        Orchestrates the analysis process: preprocess, calculate, visualize.
        """
        self._reset_analysis_data()  # Ensure clean state for processed data

        try:
            # Step 1: Preprocess (includes grayscale conversion now)
            self.preprocess_image()

            # Step 2: Calculate Metrics
            metrics = self._calculate_stroke_widths()

            # Step 3: Generate Visualization if requested and possible
            result = {'metrics': metrics, 'graphs': []}
            if debug:
                if self.skeleton is not None and self.binary_image is not None:
                    result['graphs'] = self._generate_visualization(metrics)
                elif 'error' not in metrics:
                    print("Warning: Skipping visualization due to missing intermediate data (skeleton/binary image).")

            #Preprocess Image to base64
            _, buffer = cv2.imencode('.png', self.binary_image.astype(np.uint8)*255)
            preprocessed_image_base64 = base64.b64encode(buffer).decode('utf-8')
            result['preprocessed_image'] = preprocessed_image_base64

        except Exception as e:
            print(f"An error occurred during the analysis pipeline: {e}")
            result = {
                'metrics': {k: 0.0 for k in ['mean_width', 'width_std', 'width_ratio', 'variation_coefficient']},
                'graphs': [],
                'preprocessed_image': ''
            }

        return result

# === Example Usage ===
if __name__ == '__main__':
    image_path = r"C:\Users\Samson\Desktop\Coding\IPPR\NoteMercy_Extension\backend\atest\shorthand.jpg"  # <-- CHANGE THIS PATH if needed
    analyzer = StrokeWidthAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    # print(results)

    # Print metrics in a readable format
    print("\n===== Aspect Ratio Analysis Results =====")
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
    print("\nPreprocessed Image Base64:")
    print(results['preprocessed_image'])