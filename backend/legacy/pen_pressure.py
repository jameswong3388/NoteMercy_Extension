import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class PenPressureAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the PenPressureAnalyzer with either a base64 encoded image or image path.

        Parameters:
            image_input (str): Either base64 encoded image string or image file path
            is_base64 (bool): If True, image_input is treated as base64 string, else as file path
        """
        if is_base64:
            # Decode base64 image
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            self.img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if self.img is None:
                raise ValueError("Error: Could not decode base64 image")
        else:
            # Read image from file path
            self.img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
            if self.img is None:
                raise ValueError(f"Error: Could not read image at {image_input}")

        # Convert to grayscale if necessary
        if len(self.img.shape) == 3:
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.img

        self.binary = None
        self.dist_transform = None
        self.thickness_values = None
        self.results = {}

    def _process_image(self):
        """Processes the image to compute stroke thickness measures."""
        # Apply binary thresholding with inversion
        ret, self.binary = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Calculate the distance transform
        self.dist_transform = cv2.distanceTransform(self.binary, cv2.DIST_L2, 5)

        # Get stroke thickness measurements from non-zero pixels
        self.thickness_values = self.dist_transform[self.binary > 0]

        if self.thickness_values.size == 0:
            self.results = {
                'mean_thickness': 0,
                'thickness_std': 0,
                'thickness_variance': 0,
                'coefficient_of_variation': 0
            }
        else:
            mean_thickness = np.mean(self.thickness_values)
            thickness_std = np.std(self.thickness_values)
            thickness_variance = np.var(self.thickness_values)
            coefficient_of_variation = thickness_std / mean_thickness

            self.results = {
                'mean_thickness': mean_thickness,
                'thickness_std': thickness_std,
                'thickness_variance': thickness_variance,
                'coefficient_of_variation': coefficient_of_variation
            }

    def _generate_visualization(self):
        """Generates and returns base64 encoded visualization plots."""
        plt.figure("Pen Pressure Analysis", figsize=(10, 8))

        # Original image display
        plt.subplot(2, 2, 1)
        if len(self.img.shape) == 3:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:
            plt.imshow(self.img, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        # Binary image display
        plt.subplot(2, 2, 2)
        plt.imshow(self.binary, cmap='gray')
        plt.title("Binary Image")
        plt.axis('off')

        # Distance transform visualization
        plt.subplot(2, 2, 3)
        plt.imshow(self.dist_transform, cmap='jet')
        plt.colorbar()
        plt.title("Distance Transform (Stroke Thickness)")
        plt.axis('off')

        # Histogram of thickness values
        plt.subplot(2, 2, 4)
        plt.hist(self.thickness_values, bins=30)
        plt.axvline(self.results['mean_thickness'], color='r', linestyle='--', linewidth=2)
        plt.title(
            f"Thickness Distribution\nMean: {self.results['mean_thickness']:.2f}, CoV: {self.results['coefficient_of_variation']:.2f}")
        plt.xlabel("Thickness")
        plt.ylabel("Frequency")

        plt.tight_layout()

        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return plot_base64

    def _generate_preprocessed_image_base64(self):
        """Generates and returns base64 encoded preprocessed image."""
        _, buffer = cv2.imencode('.png', self.binary)
        preprocessed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        return preprocessed_image_base64

    def analyze(self, debug=False):
        """
        Executes the complete analysis and optionally generates visualizations.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Dictionary containing metrics and optional visualization graphs.
        """
        self._process_image()

        result = {
            'metrics': self.results,
            'graphs': [],
            'preprocessed_image': self._generate_preprocessed_image_base64()
        }

        if debug:
            plot_base64 = self._generate_visualization()
            result['graphs'].append(plot_base64)

        return result


# === Example Usage ===
if __name__ == '__main__':
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png'
    analyzer = PenPressureAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results)