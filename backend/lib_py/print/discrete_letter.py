import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb


class DiscreteLetterComponentAnalyzer:
    def __init__(self, image_path):
        """
        Initialize with the path to the image.
        """
        self.image_path = image_path
        self.image = self._load_image()

    def _load_image(self):
        """
        Loads the image from the given path.
        """
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"Error: Could not read image at {self.image_path}")
            return None
        # Convert from BGR (OpenCV default) to RGB.
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def compute_components(self, debug=False):
        """
        Compute discrete letter components in the image.

        Parameters:
            debug (bool): If True, displays intermediate steps and outputs debugging info.

        Returns:
            dict: Contains number of valid letter components, average area of valid components,
                  and the total number of connected components.
        """
        if self.image is None:
            return {}

        # Convert to grayscale if the image is in color.
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            gray = self.image

        # Apply binary thresholding using Otsu's method.
        thresh = threshold_otsu(gray)
        binary = gray > thresh
        # Invert to match typical text representation (text as white on a dark background).
        binary = np.logical_not(binary)

        # Label connected components.
        labeled_image = label(binary)
        regions = regionprops(labeled_image)

        # Calculate the areas of all regions.
        areas = [region.area for region in regions]
        if len(areas) == 0:
            median_area = 0
        else:
            median_area = np.median(areas)

        # Filter out noise: consider regions with area > 25% of the median as valid.
        valid_regions = [region for region in regions if region.area > 0.25 * median_area]

        num_components = len(valid_regions)
        avg_component_area = np.mean([region.area for region in valid_regions]) if valid_regions else 0

        results = {
            'num_letter_components': num_components,
            'avg_component_area': avg_component_area,
            'total_components': len(regions)
        }

        # Debug visualization if requested.
        if debug:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # Original Image.
            axs[0, 0].imshow(self.image)
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')

            # Binary Image.
            axs[0, 1].imshow(binary, cmap='gray')
            axs[0, 1].set_title('Binary Image')
            axs[0, 1].axis('off')

            # Labeled Components (overlay).
            rgb_label = label2rgb(labeled_image, image=self.image, bg_label=0, kind='overlay')
            axs[1, 0].imshow(rgb_label)
            axs[1, 0].set_title('Connected Components')
            axs[1, 0].axis('off')

            # Valid components highlighted.
            axs[1, 1].imshow(self.image)
            for region in valid_regions:
                # region.bbox returns (min_row, min_col, max_row, max_col)
                minr, minc, maxr, maxc = region.bbox
                rect_width = maxc - minc
                rect_height = maxr - minr
                rect = plt.Rectangle((minc, minr), rect_width, rect_height,
                                     edgecolor='g', facecolor='none', linewidth=2)
                axs[1, 1].add_patch(rect)
            axs[1, 1].set_title(f'Valid Letters Found: {num_components}')
            axs[1, 1].axis('off')

            plt.tight_layout()
            plt.show()

            print(f'Number of letter components: {num_components}')
            print(f'Total components detected: {len(regions)}')
            print(f'Average component area: {avg_component_area:.2f} pixels')

        return results


# === Example Usage ===
if __name__ == '__main__':
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/4.png'
    analyzer = DiscreteLetterComponentAnalyzer(image_path)
    results = analyzer.compute_components(debug=True)
    print(results)
