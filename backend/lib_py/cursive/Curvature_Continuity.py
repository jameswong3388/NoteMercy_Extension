import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import base64
from io import BytesIO


class CursiveCurvatureAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the CursiveCurvatureAnalyzer with either a base64 encoded image or image path.

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
        self.contours = None
        self.all_polys = []
        self.segment_lengths = []

    def _preprocess_image(self):
        """
        Convert the image to grayscale and apply binary thresholding.
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, self.binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    def _find_contours(self):
        """
        Find external contours of the binary image.
        """
        self.contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def _compute_contours(self):
        """
        Process each contour to compute simplified polygon approximations and segment lengths.
        """
        for cnt in self.contours:
            # Skip very small contours that might be noise
            if cv2.contourArea(cnt) < 50:
                continue

            # Approximate contour with tolerance proportional to its perimeter
            epsilon = 0.01 * cv2.arcLength(cnt, closed=True)
            poly = cv2.approxPolyDP(cnt, epsilon, closed=True)
            self.all_polys.append(poly)

            # Calculate lengths of each segment in the polyline
            for i in range(len(poly)):
                x1, y1 = poly[i][0]
                x2, y2 = poly[(i + 1) % len(poly)][0]  # wrap-around for closed contour
                seg_len = math.hypot(x2 - x1, y2 - y1)
                self.segment_lengths.append(seg_len)

    def _calculate_statistics(self):
        """
        Normalize segment lengths by image height and compute statistical measures.
        """
        if not self.segment_lengths:
            return {
                "avg_normalized_segment_length": 0,
                "median_normalized_segment_length": 0,
                "segment_count": 0,
                "total_contours": 0
            }

        H = self.binary.shape[0]
        normalized_segment_lengths = [length / H for length in self.segment_lengths]
        avg_length = np.mean(normalized_segment_lengths)
        median_length = np.median(normalized_segment_lengths)

        return {
            "avg_normalized_segment_length": avg_length,
            "median_normalized_segment_length": median_length,
            "segment_count": len(self.segment_lengths),
            "total_contours": len(self.contours)
        }

    def _generate_visualization(self, normalized_segment_lengths, metrics):
        """
        Generate visualization plots and return them as base64 encoded images.
        """
        graphs = []

        # Draw contours and the simplified polygons on a copy of the original image
        debug_img = self.img.copy()
        cv2.drawContours(debug_img, self.contours, -1, (0, 255, 0), 2)
        cv2.drawContours(debug_img, self.all_polys, -1, (0, 0, 255), 2)

        # Create the visualization plot
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(self.binary, cmap='gray')
        plt.title("Binary Image")
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Contours (green) and Simplified Polygon (red)")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.hist(normalized_segment_lengths, bins=20)
        plt.axvline(metrics["avg_normalized_segment_length"], color='r', linestyle='dashed', linewidth=2)
        plt.title(f"Segment Length Distribution\nAvg: {metrics['avg_normalized_segment_length']:.3f}")
        plt.xlabel("Normalized Length")
        plt.ylabel("Frequency")

        plt.tight_layout()

        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        graphs.append(plot_base64)
        return graphs

    def analyze(self, debug=False):
        """
        Run the complete analysis pipeline and return metrics and optional visualizations.

        Parameters:
            debug (bool): If True, includes visualization plots in the results.

        Returns:
            dict: Contains metrics and optional visualization graphs
        """
        self._preprocess_image()
        self._find_contours()
        self._compute_contours()
        metrics = self._calculate_statistics()

        result = {
            'metrics': metrics,
            'graphs': [],
            'preprocessed_image': ''
        }

        if debug:
            _, buffer = cv2.imencode('.png', self.binary)
            preprocessed_image_base64 = base64.b64encode(buffer).decode('utf-8')
            result['preprocessed_image'] = preprocessed_image_base64

        if debug and self.segment_lengths:
            H = self.binary.shape[0]
            normalized_segment_lengths = [length / H for length in self.segment_lengths]
            result['graphs'] = self._generate_visualization(normalized_segment_lengths, metrics)

        return result


if __name__ == "__main__":
    # Example with file path
    image_path = "atest/5.png"
    analyzer = CursiveCurvatureAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results)