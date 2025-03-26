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
        # Apply binary thresholding and invert the image
        ret, binary = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)

        # Label connected components (letters/characters)
        label_img = measure.label(binary, connectivity=2)
        regions = measure.regionprops(label_img, intensity_image=self.gray_img)

        # Filter out small noise components using a minimum area threshold
        min_area = 20
        valid_regions = [region for region in regions if region.area > min_area]

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

        # Initialize lists to store metrics for each valid letter
        heights = []
        widths = []
        aspect_ratios = []
        mean_intensities = []
        intensity_variations = []
        stroke_widths = []
        bboxes = []  # For debug visualization (bounding boxes)
        centroids = []  # For debug visualization (centroid coordinates)

        # Process each valid region
        for i, region in enumerate(valid_regions):
            # Get bounding box (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = region.bbox
            width = max_col - min_col
            height = max_row - min_row
            # Avoid division by zero
            aspect_ratio = width / height if height != 0 else 0
            heights.append(height)
            widths.append(width)
            aspect_ratios.append(aspect_ratio)
            # Store bounding box as (x, y, width, height) and centroid (x, y)
            bboxes.append((min_col, min_row, width, height))
            centroids.append((region.centroid[1], region.centroid[0]))

            # Extract intensity values for this letter (inverting intensity so higher value = darker)
            coords = region.coords
            # Note: region.coords is an array of (row, col) coordinates.
            letter_intensity = 255 - self.gray_img[coords[:, 0], coords[:, 1]].astype(np.float64)
            mean_intensity = np.mean(letter_intensity)
            std_intensity = np.std(letter_intensity)
            mean_intensities.append(mean_intensity)
            intensity_variations.append(std_intensity)

            # Estimate stroke width using distance transform on the letter mask
            mask = np.zeros(binary.shape, dtype=np.uint8)
            mask[coords[:, 0], coords[:, 1]] = 1
            dist_transform = distance_transform_edt(mask)
            max_dist = np.max(dist_transform[mask == 1])
            stroke_width = 2 * max_dist  # Double the maximum distance to approximate stroke width
            stroke_widths.append(stroke_width)

        # Convert lists to numpy arrays for statistical computations
        heights = np.array(heights)
        widths = np.array(widths)
        aspect_ratios = np.array(aspect_ratios)
        mean_intensities = np.array(mean_intensities)
        stroke_widths = np.array(stroke_widths)

        # Compute coefficient of variation (cv = std/mean) and then uniformity as (1 - cv)
        height_cv = np.std(heights) / np.mean(heights)
        width_cv = np.std(widths) / np.mean(widths)
        aspect_ratio_cv = np.std(aspect_ratios) / np.mean(aspect_ratios)
        pressure_cv = np.std(mean_intensities) / np.mean(mean_intensities)
        stroke_width_cv = np.std(stroke_widths) / np.mean(stroke_widths)

        height_uniformity = max(0, 1 - height_cv)
        width_uniformity = max(0, 1 - width_cv)
        aspect_ratio_uniformity = max(0, 1 - aspect_ratio_cv)
        pen_pressure_uniformity = max(0, 1 - pressure_cv)
        stroke_width_uniformity = max(0, 1 - stroke_width_cv)

        # Pack results into a metrics dictionary
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

        # Debug visualization if requested
        if debug:
            plt.figure("Letter Size and Pen Pressure Analysis", figsize=(12, 8))

            # Original Image
            plt.subplot(2, 3, 1)
            # Convert BGR to RGB for proper display
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
    image_path = '/path/to/your/image.png'
    analyzer = LetterUniformityAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])
