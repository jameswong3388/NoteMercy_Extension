import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


class StrokeWidthAnalyzer:
    def __init__(self, debug=False):
        """
        Initializes the analyzer.

        Parameters:
            debug (bool): If True, shows intermediate plots and prints debug info.
        """
        self.debug = debug

    def compute_stroke_width_variation(self, image_path):
        """
        Computes stroke width variation metrics for an image.

        Parameters:
            image_path (str): Path to the input image.

        Returns:
            dict: A dictionary with mean_width, width_std, width_ratio,
                  and variation_coefficient.
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return {}

        # Convert to grayscale if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Binarize the image using a threshold of 127 (scale: 0-255)
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # Invert image so that text becomes white (foreground) on a black background
        bw = cv2.bitwise_not(bw)
        # Convert to boolean for further processing (True for text)
        bw_bool = bw > 0

        # Compute the distance transform on the inverted binary image.
        # In MATLAB: D = bwdist(~bw); here ~bw_bool gives the background.
        D = distance_transform_edt(~bw_bool)

        # Compute skeleton of the text strokes
        # skimage's skeletonize expects a boolean image with True as foreground.
        skel = skeletonize(bw_bool)

        # Get stroke radii from the distance transform at skeleton points
        stroke_radii = D[skel]
        # Remove zeros (background)
        stroke_radii = stroke_radii[stroke_radii > 0]

        if stroke_radii.size == 0:
            return {
                'mean_width': 0,
                'width_std': 0,
                'width_ratio': 0,
                'variation_coefficient': 0
            }

        # Calculate stroke width metrics
        # Multiply by 2 to approximate full stroke width (from radius)
        mean_width = 2 * np.mean(stroke_radii)
        width_std = 2 * np.std(stroke_radii)
        width_ratio = np.max(stroke_radii) / np.min(stroke_radii)
        variation_coefficient = width_std / mean_width

        results = {
            'mean_width': mean_width,
            'width_std': width_std,
            'width_ratio': width_ratio,
            'variation_coefficient': variation_coefficient
        }

        # Debug visualization if requested
        if self.debug:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Stroke Width Analysis', fontsize=16)

            # Original image (convert BGR to RGB for displaying)
            axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')

            # Binary image
            axs[0, 1].imshow(bw, cmap='gray')
            axs[0, 1].set_title('Binary Image')
            axs[0, 1].axis('off')

            # Skeleton overlay on binary image
            axs[1, 0].imshow(bw, cmap='gray')
            # Get coordinates where skeleton is True
            y_coords, x_coords = np.nonzero(skel)
            axs[1, 0].scatter(x_coords, y_coords, s=1, c='r')
            axs[1, 0].set_title('Skeleton Points')
            axs[1, 0].axis('off')

            # Histogram of stroke widths
            widths = 2 * stroke_radii
            axs[1, 1].hist(widths, bins=20, edgecolor='black')
            axs[1, 1].axvline(mean_width, color='r', linestyle='--', linewidth=2)
            axs[1, 1].set_title(f'Stroke Width Distribution\nMean: {mean_width:.2f}, Ratio: {width_ratio:.2f}')
            axs[1, 1].set_xlabel('Stroke Width')
            axs[1, 1].set_ylabel('Frequency')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

            # Print debug information
            print(f"Mean stroke width: {mean_width:.3f}")
            print(f"Stroke width std: {width_std:.3f}")
            print(f"Thick-thin ratio: {width_ratio:.3f}")
            print(f"Variation coefficient: {variation_coefficient:.3f}")

        return results


# === Example Usage ===
if __name__ == '__main__':
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png'  # Update this path accordingly
    analyzer = StrokeWidthAnalyzer(debug=True)
    results = analyzer.compute_stroke_width_variation(image_path)
    print(results)
