import cv2
import numpy as np
import matplotlib.pyplot as plt


class PenPressureAnalyzer:
    def __init__(self, image_path, debug=False):
        """
        Initialize the analyzer with the image path and debug flag.

        Args:
            image_path (str): Path to the image file.
            debug (bool): If True, shows debug plots.
        """
        self.image_path = image_path
        self.debug = debug
        self.img = None
        self.gray = None
        self.binary = None
        self.dist_transform = None
        self.thickness_values = None
        self.results = {}

    def load_image(self):
        """Loads the image from the specified path."""
        self.img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if self.img is None:
            raise ValueError(f"Error: Could not read image at {self.image_path}")

    def process_image(self):
        """Processes the image to compute stroke thickness measures."""
        # Convert to grayscale if necessary.
        if len(self.img.shape) == 3:
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.img

        # Apply binary thresholding with inversion.
        ret, self.binary = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Calculate the distance transform.
        self.dist_transform = cv2.distanceTransform(self.binary, cv2.DIST_L2, 5)

        # Get stroke thickness measurements from non-zero pixels.
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

    def debug_visualization(self):
        """Displays the debug visualizations of the analysis."""
        plt.figure("Pen Pressure Analysis", figsize=(10, 8))

        # Original image display.
        plt.subplot(2, 2, 1)
        if len(self.img.shape) == 3:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:
            plt.imshow(self.img, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        # Binary image display.
        plt.subplot(2, 2, 2)
        plt.imshow(self.binary, cmap='gray')
        plt.title("Binary Image")
        plt.axis('off')

        # Distance transform visualization.
        plt.subplot(2, 2, 3)
        plt.imshow(self.dist_transform, cmap='jet')
        plt.colorbar()
        plt.title("Distance Transform (Stroke Thickness)")
        plt.axis('off')

        # Histogram of thickness values.
        plt.subplot(2, 2, 4)
        plt.hist(self.thickness_values, bins=30)
        plt.axvline(self.results['mean_thickness'], color='r', linestyle='--', linewidth=2)
        plt.title(
            f"Thickness Distribution\nMean: {self.results['mean_thickness']:.2f}, CoV: {self.results['coefficient_of_variation']:.2f}")
        plt.xlabel("Thickness")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

        # Print debug information.
        print(f"Mean stroke thickness: {self.results.get('mean_thickness', 0):.3f}")
        print(f"Thickness standard deviation: {self.results.get('thickness_std', 0):.3f}")
        print(f"Coefficient of variation: {self.results.get('coefficient_of_variation', 0):.3f}")

    def analyze(self):
        """
        Executes the complete analysis: load image, process it, and optionally visualize the results.

        Returns:
            dict: Dictionary containing the computed statistics.
        """
        self.load_image()
        self.process_image()
        if self.debug:
            self.debug_visualization()
        return self.results


# === Example Usage ===
if __name__ == '__main__':
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png'
    analyzer = PenPressureAnalyzer(image_path, debug=True)
    pressure_results = analyzer.analyze()
    print(pressure_results)
