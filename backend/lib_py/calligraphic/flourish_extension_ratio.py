import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

class FlourishAnalyzer:
    def __init__(self, image_path, debug=False):
        self.image_path = image_path
        self.debug = debug
        self.img = None
        self.gray = None
        self.binary = None

    def read_and_preprocess(self):
        # Read the image
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            print(f"Error: Could not read image at {self.image_path}")
            return False

        # Convert to grayscale if the image is RGB
        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.img

        # Binarize using Otsu thresholding
        thresh = threshold_otsu(self.gray)
        binary = self.gray > thresh
        # Invert the binary image (to mimic MATLAB's inversion)
        self.binary = np.logical_not(binary)
        return True

    def compute_metrics(self):
        # Label connected components and extract region properties
        labeled = label(self.binary)
        props = regionprops(labeled)
        if len(props) == 0:
            return ({
                'flourish_ratio': 0,
                'width_ratio': 0,
                'height_ratio': 0,
                'core_area': 0,
                'complexity_ratio': 0,
                'vertical_proportion': 0
            }, None, None, None)

        # Find the main letter body (largest connected component)
        main_component = max(props, key=lambda r: r.area)
        minr, minc, maxr, maxc = main_component.bbox
        core_width = maxc - minc
        core_height = maxr - minr
        core_area = main_component.area

        # Compute the overall bounding box (total extent)
        coords = np.column_stack(np.where(self.binary))
        if coords.size == 0:
            total_width = 0
            total_height = 0
            total_box = (0, 0, 0, 0)
        else:
            min_total = coords.min(axis=0)  # (row, col)
            max_total = coords.max(axis=0)
            total_height = max_total[0] - min_total[0]
            total_width = max_total[1] - min_total[1]
            # For drawing, convert to (x, y, width, height)
            total_box = (min_total[1], min_total[0], total_width, total_height)

        width_ratio = total_width / core_width if core_width else 0
        height_ratio = total_height / core_height if core_height else 0
        flourish_ratio = (width_ratio + height_ratio) / 2

        # Calculate contour complexity using perimeter-to-area ratio
        total_perimeter = sum(r.perimeter for r in props)
        total_area = sum(r.area for r in props)
        complexity_ratio = (total_perimeter**2 / total_area) if total_area > 0 else 0

        # Calculate vertical distribution metrics
        vertical_proj = np.sum(self.binary, axis=1)  # Sum along columns
        total_proj = np.sum(vertical_proj)
        if total_proj > 0:
            cum_dist = np.cumsum(vertical_proj) / total_proj
        else:
            cum_dist = np.zeros_like(vertical_proj)

        # Find indices corresponding to the 5th and 95th percentiles
        low_indices = np.where(cum_dist > 0.05)[0]
        high_indices = np.where(cum_dist < 0.95)[0]
        # Adjust indices to mimic MATLAB's 1-indexing in the original code
        low_quant = low_indices[0] + 1 if low_indices.size > 0 else 0
        high_quant = high_indices[-1] + 1 if high_indices.size > 0 else self.binary.shape[0]

        vertical_proportion = (self.binary.shape[0] - high_quant - low_quant) / self.binary.shape[0]

        results = {
            'flourish_ratio': flourish_ratio,
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'core_area': core_area,
            'complexity_ratio': complexity_ratio,
            'vertical_proportion': vertical_proportion
        }
        # For debugging, we return:
        # - vertical_proj: the horizontal projection of pixels
        # - main_box: bounding box for the main component in (x, y, width, height) format
        # - total_box: overall bounding box (x, y, width, height)
        main_box = (minc, minr, core_width, core_height)
        return results, vertical_proj, main_box, total_box

    def debug_plot(self, results, vertical_proj, main_box, total_box):
        # Unpack the main bounding box (x, y, width, height)
        x_main, y_main, width_main, height_main = main_box
        # Unpack the total bounding box (x, y, width, height)
        x_total, y_total, total_width, total_height = total_box

        plt.figure(figsize=(12, 8))

        # Subplot 1: Original Image
        plt.subplot(2, 3, 1)
        # Convert BGR to RGB if needed
        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:
            plt.imshow(self.img, cmap='gray')
        plt.title('Original Image')

        # Subplot 2: Binary Image
        plt.subplot(2, 3, 2)
        plt.imshow(self.binary, cmap='gray')
        plt.title('Binary Image')

        # Subplot 3: Bounding Boxes on Binary Image
        plt.subplot(2, 3, 3)
        plt.imshow(self.binary, cmap='gray')
        ax = plt.gca()
        # Draw main component bounding box in red
        rect_main = plt.Rectangle((x_main, y_main), width_main, height_main,
                                  edgecolor='r', facecolor='none', linewidth=2)
        ax.add_patch(rect_main)
        # Draw total bounding box in blue
        rect_total = plt.Rectangle((x_total, y_total), total_width, total_height,
                                   edgecolor='b', facecolor='none', linewidth=2)
        ax.add_patch(rect_total)
        plt.title('Bounding Boxes')

        # Subplot 4: Vertical Distribution
        plt.subplot(2, 3, 4)
        plt.plot(vertical_proj, np.arange(len(vertical_proj)))
        plt.title('Vertical Distribution')
        plt.xlabel('Pixel Count')
        plt.ylabel('Vertical Position')

        # Subplot 5 and 6 combined: Metrics Display
        plt.subplot(2, 3, (5, 6))
        plt.axis('off')
        text_str = (
            f"Flourish Ratio: {results['flourish_ratio']:.2f}\n"
            f"Width Ratio: {results['width_ratio']:.2f}\n"
            f"Height Ratio: {results['height_ratio']:.2f}\n"
            f"Core Area: {results['core_area']} pixels\n"
            f"Complexity Ratio: {results['complexity_ratio']:.2f}\n"
            f"Vertical Proportion: {results['vertical_proportion']:.2f}"
        )
        plt.text(0.1, 0.5, text_str, fontsize=12, transform=plt.gca().transAxes)
        plt.title('Metrics')

        plt.tight_layout()
        plt.show()

        # Also print out debug information
        print("\nCalligraphic Analysis Results:")
        print(f"Flourish extension ratio: {results['flourish_ratio']:.2f}")
        print(f"Complexity ratio: {results['complexity_ratio']:.2f}")
        print(f"Vertical proportion: {results['vertical_proportion']:.2f}")

    def analyze(self):
        if not self.read_and_preprocess():
            return {}
        results, vertical_proj, main_box, total_box = self.compute_metrics()
        if self.debug and vertical_proj is not None:
            self.debug_plot(results, vertical_proj, main_box, total_box)
        return results

# === Example Usage ===
if __name__ == "__main__":
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/4.png'
    analyzer = FlourishAnalyzer(image_path, debug=True)
    flourish_results = analyzer.analyze()
    print(flourish_results)
