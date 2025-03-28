import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import distance_transform_edt
import base64
from io import BytesIO

class LetterUniformityAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the LetterUniformityAnalyzer with either a base64 encoded image or image path.

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
        Analyzes the image to determine letter size uniformity by measuring
        height, width, aspect ratio, pen pressure, and stroke width.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Metrics and visualization graphs if debug=True.
        """
        # -----------------------------
        # 1) Apply Otsu's thresholding
        # -----------------------------
        # Using Otsu automatically determines the best threshold for binarization
        # in images with bimodal histograms.
        blur = cv2.GaussianBlur(self.gray_img, (3, 3), 0)
        # Otsu’s threshold returns the threshold value as 'thresh_val'
        thresh_val, binary = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # ---------------------------------------
        # 2) Morphological operations (optional)
        # ---------------------------------------
        # A small opening removes noise. Adjust kernel size as needed.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Optionally, a small closing can unify broken strokes
        # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # ---------------------------------------------------
        # 3) Label connected components (letters/characters)
        # ---------------------------------------------------
        label_img = measure.label(binary, connectivity=2)
        regions = measure.regionprops(label_img, intensity_image=self.gray_img)

        # --------------------------------------------------------------------------------
        # 4) Filter out small noise components using a dynamic minimum area threshold
        # --------------------------------------------------------------------------------
        # For example, we can compute a robust estimate (like median area) among all
        # connected components, then set some fraction of that as the threshold.
        all_areas = [r.area for r in regions]
        median_area = np.median(all_areas) if all_areas else 0
        # As a starting point, we set min_area to 0.2 * median_area, but at least 20.
        min_area = max(20, 0.2 * median_area)

        valid_regions = [region for region in regions if region.area >= min_area]

        # If no regions were found, return zeros
        if len(valid_regions) == 0:
            metrics = {
                'height_uniformity': 0,
                'width_uniformity': 0,
                'aspect_ratio_uniformity': 0,
                'pen_pressure_uniformity': 0,
                'stroke_width_uniformity': 0,
                'avg_pen_pressure': 0,
                'avg_stroke_width': 0,
                'letter_count': 0
            }
            return {'metrics': metrics, 'graphs': []}

        # ---------------------------------------------------------------------------
        # 5) Compute metrics (height, width, aspect ratio, stroke width, pen pressure)
        # ---------------------------------------------------------------------------
        heights = []
        widths = []
        aspect_ratios = []
        mean_intensities = []
        stroke_widths = []
        bboxes = []
        centroids = []

        for i, region in enumerate(valid_regions):
            # Get bounding box (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = region.bbox
            width = max_col - min_col
            height = max_row - min_row
            aspect_ratio = width / height if height != 0 else 0

            heights.append(height)
            widths.append(width)
            aspect_ratios.append(aspect_ratio)
            bboxes.append((min_col, min_row, width, height))
            centroids.append((region.centroid[1], region.centroid[0]))

            # Extract intensity values for this letter (invert so higher = darker)
            coords = region.coords
            letter_intensity = 255 - self.gray_img[coords[:, 0], coords[:, 1]].astype(np.float64)
            mean_intensity = np.mean(letter_intensity)
            mean_intensities.append(mean_intensity)

            # Estimate stroke width using distance transform
            mask = np.zeros(binary.shape, dtype=np.uint8)
            mask[coords[:, 0], coords[:, 1]] = 1
            dist_transform = distance_transform_edt(mask)
            max_dist = np.max(dist_transform[mask == 1])
            stroke_width = 2 * max_dist
            stroke_widths.append(stroke_width)

        # ---------------------------------------------------------------------------
        # 6) Compute uniformities using coefficient of variation (CV = std / mean)
        # ---------------------------------------------------------------------------
        def uniformity_score(values):
            if len(values) == 0 or np.mean(values) == 0:
                return 0
            cv = np.std(values) / np.mean(values)
            return max(0, 1 - cv)  # ensure we don't return negative

        height_uniformity = uniformity_score(heights)
        width_uniformity = uniformity_score(widths)
        aspect_ratio_uniformity = uniformity_score(aspect_ratios)
        pen_pressure_uniformity = uniformity_score(mean_intensities)
        stroke_width_uniformity = uniformity_score(stroke_widths)

        metrics = {
            'height_uniformity': height_uniformity,
            'width_uniformity': width_uniformity,
            'aspect_ratio_uniformity': aspect_ratio_uniformity,
            'pen_pressure_uniformity': pen_pressure_uniformity,
            'stroke_width_uniformity': stroke_width_uniformity,
            'avg_pen_pressure': np.mean(mean_intensities) / 255,
            'avg_stroke_width': np.mean(stroke_widths),
            'letter_count': len(valid_regions)
        }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # -----------------------------------------------------
        # 7) Optional Debug Visualization
        # -----------------------------------------------------
        if debug:
            plt.figure("Letter Size and Pen Pressure Analysis", figsize=(12, 8))

            # Original Image
            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            # Binary Image with Bounding Boxes
            plt.subplot(2, 3, 2)
            plt.imshow(binary, cmap='gray')
            ax = plt.gca()
            for idx, (bbox, centroid) in enumerate(zip(bboxes, centroids)):
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     edgecolor='r', facecolor='none', linewidth=1)
                ax.add_patch(rect)
                plt.text(centroid[0], centroid[1], str(idx + 1),
                         color='g', fontsize=8, fontweight='bold')
            plt.title("Letter Detection")
            plt.axis('off')

            # Height Distribution
            plt.subplot(2, 3, 3)
            bins = min(20, int(np.ceil(len(heights) / 2)) or 1)
            plt.hist(heights, bins=bins)
            plt.title(f"Height Distribution\nUniformity: {height_uniformity:.3f}")

            # Width Distribution
            plt.subplot(2, 3, 4)
            bins = min(20, int(np.ceil(len(widths) / 2)) or 1)
            plt.hist(widths, bins=bins)
            plt.title(f"Width Distribution\nUniformity: {width_uniformity:.3f}")

            # Pen Pressure Distribution
            plt.subplot(2, 3, 5)
            bins = min(20, int(np.ceil(len(mean_intensities) / 2)) or 1)
            plt.hist(mean_intensities, bins=bins)
            plt.title(f"Pen Pressure Distribution\nUniformity: {pen_pressure_uniformity:.3f}")

            # Stroke Width Distribution
            plt.subplot(2, 3, 6)
            bins = min(20, int(np.ceil(len(stroke_widths) / 2)) or 1)
            plt.hist(stroke_widths, bins=bins)
            plt.title(f"Stroke Width Distribution\nUniformity: {stroke_width_uniformity:.3f}")

            plt.tight_layout()

            # Convert plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            result['graphs'].append(plot_base64)

        return result

# === Example usage ===
if __name__ == "__main__":
    # Example with file path
    image_path = '../../atest/print.jpg'
    analyzer = LetterUniformityAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])
