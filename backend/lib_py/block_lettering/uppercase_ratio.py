import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


class UppercaseRatioAnalyzer:
    def __init__(self, image_path, debug=False):
        """
        Initialize the analyzer with an image path and debug flag.
        """
        self.image_path = image_path
        self.debug = debug
        self.img = None
        self.gray = None
        self.binary = None
        self.results = None
        self._load_image()

    def _load_image(self):
        """
        Loads the image and converts it to grayscale if needed.
        """
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise ValueError(f"Error: Could not read image at {self.image_path}")
        # Convert to grayscale if image is in color
        if len(self.img.shape) == 3:
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.img

    def analyze(self):
        """
        Analyzes the image to calculate the ratio of uppercase-like characters.
        Returns a dictionary with the results.
        """
        # Apply binary thresholding and invert
        _, binary = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY)
        self.binary = cv2.bitwise_not(binary)  # Inversion to match typical document analysis

        # Convert binary image to boolean for regionprops
        binary_bool = self.binary > 0

        # Find connected components (potential characters)
        labeled_img = label(binary_bool)
        props = regionprops(labeled_img)

        if len(props) == 0:
            self.results = {
                'uppercase_ratio': 0,
                'character_count': 0
            }
            return self.results

        # Filter out noise by area: keep components larger than 10% of the median area
        areas = [prop.area for prop in props]
        median_area = np.median(areas)
        filtered_props = [prop for prop in props if prop.area > (median_area * 0.1)]
        if len(filtered_props) == 0:
            self.results = {
                'uppercase_ratio': 0,
                'character_count': 0
            }
            return self.results

        # Extract bounding box heights
        heights = [prop.bbox[2] - prop.bbox[0] for prop in filtered_props]
        median_height = np.median(heights)
        normalized_heights = [h / median_height for h in heights]

        # Total image height for reference
        img_height = self.binary.shape[0]

        # Determine uppercase/lowercase classification threshold
        uppercase_threshold = 0.8  # Characters with normalized height >= 0.8 are considered uppercase-like
        uppercase_count = sum(1 for nh in normalized_heights if nh >= uppercase_threshold)
        total_chars = len(normalized_heights)
        uppercase_ratio = uppercase_count / total_chars if total_chars > 0 else 0

        # Calculate extents (ratio of component pixels to bounding box pixels)
        extents = [prop.extent for prop in filtered_props]
        median_extent = np.median(extents)

        self.results = {
            'uppercase_ratio': uppercase_ratio,
            'character_count': total_chars,
            'median_height_ratio': median_height / img_height,
            'median_extent': median_extent
        }

        if self.debug:
            self._debug_visualization(filtered_props, normalized_heights, uppercase_threshold, uppercase_ratio)
            print(f"Uppercase ratio: {uppercase_ratio:.2f}")
            print(f"Character count: {total_chars}")
            print(f"Median height ratio: {self.results['median_height_ratio']:.3f}")
            print(f"Median extent: {median_extent:.3f}")

        return self.results

    def _debug_visualization(self, filtered_props, normalized_heights, uppercase_threshold, uppercase_ratio):
        """
        Displays debug visualizations including the original image,
        binary image, bounding boxes around detected characters, and a histogram
        of normalized character heights.
        """
        # Prepare figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Uppercase Ratio Analysis", fontsize=16)

        # Original Image (convert BGR to RGB for correct display)
        if len(self.img.shape) == 3:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            axs[0, 0].imshow(img_rgb)
        else:
            axs[0, 0].imshow(self.img, cmap='gray')
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')

        # Binary Image
        axs[0, 1].imshow(self.binary, cmap='gray')
        axs[0, 1].set_title('Binary Image')
        axs[0, 1].axis('off')

        # Visualize character bounding boxes on original image
        axs[1, 0].imshow(img_rgb if len(self.img.shape) == 3 else self.img, cmap='gray')
        for idx, prop in enumerate(filtered_props):
            minr, minc, maxr, maxc = prop.bbox
            width = maxc - minc
            height = maxr - minr
            # Determine color: red for uppercase, blue for lowercase
            color = 'r' if normalized_heights[idx] >= uppercase_threshold else 'b'
            rect = plt.Rectangle((minc, minr), width, height, edgecolor=color, facecolor='none', linewidth=2)
            axs[1, 0].add_patch(rect)
        axs[1, 0].set_title('Character Classification\n(Red=Uppercase, Blue=Lowercase)')
        axs[1, 0].axis('off')

        # Histogram of normalized heights
        axs[1, 1].hist(normalized_heights, bins=10, edgecolor='black')
        axs[1, 1].axvline(uppercase_threshold, color='red', linestyle='--', linewidth=2)
        axs[1, 1].set_title(f'Height Distribution\nUppercase Ratio: {uppercase_ratio:.2f}')
        axs[1, 1].set_xlabel('Normalized Height')
        axs[1, 1].set_ylabel('Frequency')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# === Example usage ===
if __name__ == '__main__':
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/3.png'
    analyzer = UppercaseRatioAnalyzer(image_path, debug=True)
    results = analyzer.analyze()
    print(results)
