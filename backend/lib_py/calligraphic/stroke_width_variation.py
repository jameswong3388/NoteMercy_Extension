import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class StrokeWidthAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the StrokeWidthAnalyzer with either a base64 encoded image or image path.

        Parameters:
            image_input (str): Either base64 encoded image string or image file path
            is_base64 (bool): If True, image_input is treated as base64 string, else as file path
        """
        if is_base64:
            # Decode base64 image
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if self.img is None:
                raise ValueError("Error: Could not decode base64 image")
        else:
            # Read image from file path
            self.img = cv2.imread(image_input)
            if self.img is None:
                raise ValueError(f"Error: Could not read image at {image_input}")

        # Convert to grayscale if needed
        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_img = self.img.copy()

    def analyze(self, debug=False):
        """
        Analyzes the image to determine stroke width variation metrics.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Metrics including mean_width, width_std, width_ratio,
                  and variation_coefficient, plus visualization graphs if debug=True.
        """
        # Binarize the image using a threshold of 127 (scale: 0-255)
        _, bw = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_BINARY)
        # Invert image so that text becomes white (foreground) on a black background
        bw = cv2.bitwise_not(bw)
        # Convert to boolean for further processing (True for text)
        bw_bool = bw > 0

        # Compute the distance transform on the inverted binary image
        D = distance_transform_edt(~bw_bool)

        # Compute skeleton of the text strokes
        skel = skeletonize(bw_bool)

        # Get stroke radii from the distance transform at skeleton points
        stroke_radii = D[skel]
        # Remove zeros (background)
        stroke_radii = stroke_radii[stroke_radii > 0]

        if stroke_radii.size == 0:
            metrics = {
                'mean_width': 0,
                'width_std': 0,
                'width_ratio': 0,
                'variation_coefficient': 0
            }
        else:
            # Calculate stroke width metrics
            mean_width = 2 * np.mean(stroke_radii)
            width_std = 2 * np.std(stroke_radii)
            width_ratio = np.max(stroke_radii) / np.min(stroke_radii)
            variation_coefficient = width_std / mean_width

            metrics = {
                'mean_width': mean_width,
                'width_std': width_std,
                'width_ratio': width_ratio,
                'variation_coefficient': variation_coefficient
            }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # Generate visualization plots if debug mode is enabled
        if debug:
            # Create figure with subplots
            plt.figure("Stroke Width Analysis", figsize=(12, 10))
            plt.suptitle('Stroke Width Analysis', fontsize=16)

            # Subplot 1: Original Image
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')

            # Subplot 2: Binary Image
            plt.subplot(2, 2, 2)
            plt.imshow(bw, cmap='gray')
            plt.title('Binary Image')
            plt.axis('off')

            # Subplot 3: Skeleton Points
            plt.subplot(2, 2, 3)
            plt.imshow(bw, cmap='gray')
            y_coords, x_coords = np.nonzero(skel)
            plt.scatter(x_coords, y_coords, s=1, c='r')
            plt.title('Skeleton Points')
            plt.axis('off')

            # Subplot 4: Stroke Width Distribution
            plt.subplot(2, 2, 4)
            widths = 2 * stroke_radii
            plt.hist(widths, bins=20, edgecolor='black')
            plt.axvline(metrics['mean_width'], color='r', linestyle='--', linewidth=2)
            plt.title(f'Stroke Width Distribution\nMean: {metrics["mean_width"]:.2f}, Ratio: {metrics["width_ratio"]:.2f}')
            plt.xlabel('Stroke Width')
            plt.ylabel('Frequency')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Convert plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            result['graphs'].append(plot_base64)

        return result


# === Example Usage ===
if __name__ == '__main__':
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png'
    analyzer = StrokeWidthAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results)
