import cv2
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO


class VerticalStrokeAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the VerticalStrokeAnalyzer with either a base64 encoded image or image path.

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
        
        self.binary = None
        self.boxes = []
        self.metrics = {}

    def _preprocess_image(self):
        """Converts image to grayscale and applies thresholding to get a binary image."""
        # Convert image to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to get a binary image (inverted)
        _, self.binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def _find_components(self):
        """Finds connected components (letters or parts of letters)."""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.binary, connectivity=8)
        # Skip background (label 0) and filter out small components (noise)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if h > 5 and w > 5:
                self.boxes.append((x, y, w, h))

    def _compute_metrics(self):
        """Computes the vertical stroke metrics including median height, max height, and ascender ratio."""
        # Extract the height of each component
        heights = [h for (x, y, w, h) in self.boxes]
        if heights:
            heights.sort()
            median_height = np.median(heights)  # Approximate x-height (middle zone)
            max_height = max(heights)  # Height of the tallest stroke (ascender or capital)
            ascender_ratio = max_height / median_height if median_height > 0 else 0

            self.metrics = {
                "median_height": float(median_height),
                "max_height": float(max_height),
                "ascender_ratio": float(ascender_ratio)
            }
        else:
            self.metrics = {
                "median_height": 0,
                "max_height": 0,
                "ascender_ratio": 0
            }

    def analyze(self, debug=False):
        """
        Analyzes the image to determine vertical stroke proportions.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Metrics including median height, max height, and ascender ratio,
                  plus visualization graphs if debug=True.
        """
        self._preprocess_image()
        self._find_components()
        self._compute_metrics()
        
        result = {
            'metrics': self.metrics,
            'graphs': []
        }
        
        # Generate visualization plots if debug mode is enabled
        if debug and self.boxes:
            # Draw bounding boxes on the original image
            img_with_boxes = self.img.copy()
            for (x, y, w, h) in self.boxes:
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Create a figure with subplots for visualization
            plt.figure(figsize=(15, 10))

            plt.subplot(131)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            plt.axis('off')

            plt.subplot(132)
            plt.title("Binary Image")
            plt.imshow(self.binary, cmap='gray')
            plt.axis('off')

            plt.subplot(133)
            plt.title(f"Bounding Boxes (Ascender Ratio: {self.metrics.get('ascender_ratio', 0):.2f})")
            plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            plt.axis('off')

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
    image_path = "atest/3.png"
    analyzer = VerticalStrokeAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])
