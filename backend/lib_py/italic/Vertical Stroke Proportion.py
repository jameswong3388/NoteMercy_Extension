import cv2
import matplotlib.pyplot as plt
import numpy as np


class VerticalStrokeAnalyzer:
    def __init__(self, image_path, debug=False):
        self.image_path = image_path
        self.debug = debug
        self.img = None
        self.binary = None
        self.boxes = []
        self.results = {}

    def load_image(self):
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise FileNotFoundError(f"Image file not found: {self.image_path}")

    def preprocess_image(self):
        # Convert image to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to get a binary image (inverted)
        _, self.binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def find_components(self):
        # Find connected components (letters or parts of letters)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.binary, connectivity=8)
        # Skip background (label 0) and filter out small components (noise)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if h > 5 and w > 5:
                self.boxes.append((x, y, w, h))

    def compute_metrics(self):
        # Extract the height of each component
        heights = [h for (x, y, w, h) in self.boxes]
        if heights:
            heights.sort()
            median_height = np.median(heights)  # Approximate x-height (middle zone)
            max_height = max(heights)  # Height of the tallest stroke (ascender or capital)
            ascender_ratio = max_height / median_height if median_height > 0 else 0

            self.results = {
                "median_height": float(median_height),
                "max_height": float(max_height),
                "ascender_ratio": float(ascender_ratio)
            }
            print(f"Ascender-to-xHeight ratio: {ascender_ratio:.2f}")
        else:
            self.results = {}

    def debug_visualization(self):
        if self.debug and self.boxes:
            # Draw bounding boxes on the original image
            img_with_boxes = self.img.copy()
            for (x, y, w, h) in self.boxes:
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Create a figure with subplots for visualization
            plt.figure(figsize=(15, 10))

            plt.subplot(131)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

            plt.subplot(132)
            plt.title("Binary Image")
            plt.imshow(self.binary, cmap='gray')

            plt.subplot(133)
            plt.title(f"Bounding Boxes (Ascender Ratio: {self.results.get('ascender_ratio', 0):.2f})")
            plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))

            plt.tight_layout()
            plt.show()

    def analyze(self):
        self.load_image()
        self.preprocess_image()
        self.find_components()
        self.compute_metrics()
        self.debug_visualization()
        return self.results


if __name__ == "__main__":
    # Replace with your actual image file
    image_path = "atest/3.png"
    analyzer = VerticalStrokeAnalyzer(image_path, debug=True)
    results = analyzer.analyze()
    print(results)
