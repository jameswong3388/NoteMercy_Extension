import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class StrokeCurvatureContinuity:
    def __init__(self, image_path):
        """
        Initialize the analyzer with the given image path.
        """
        self.image_path = image_path
        self.img = None
        self.binary = None
        self.contours = None
        self.all_polys = []
        self.segment_lengths = []
        self.results = {}

    def read_image(self):
        """
        Read the image from the file system.
        """
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise ValueError(f"Error: Could not read image at {self.image_path}")

    def preprocess_image(self):
        """
        Convert the image to grayscale and apply binary thresholding.
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, self.binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    def find_contours(self):
        """
        Find external contours of the binary image.
        """
        self.contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def compute_contours(self):
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

    def calculate_statistics(self):
        """
        Normalize segment lengths by image height and compute statistical measures.
        """
        if not self.segment_lengths:
            self.results = {"avg_normalized_segment_length": 0, "segment_count": 0}
            return

        H = self.binary.shape[0]
        normalized_segment_lengths = [length / H for length in self.segment_lengths]
        avg_length = np.mean(normalized_segment_lengths)
        median_length = np.median(normalized_segment_lengths)

        self.results = {
            "avg_normalized_segment_length": avg_length,
            "median_normalized_segment_length": median_length,
            "segment_count": len(self.segment_lengths),
            "total_contours": len(self.contours)
        }
        return normalized_segment_lengths

    def debug_visualization(self, normalized_segment_lengths):
        """
        Visualize the results including original image, binary image, contours, and segment length distribution.
        """
        # Draw contours and the simplified polygons on a copy of the original image
        debug_img = self.img.copy()
        cv2.drawContours(debug_img, self.contours, -1, (0, 255, 0), 2)
        cv2.drawContours(debug_img, self.all_polys, -1, (0, 0, 255), 2)

        # Plot the different stages
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(2, 2, 2)
        plt.imshow(self.binary, cmap='gray')
        plt.title("Binary Image")

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Contours (green) and Simplified Polygon (red)")

        plt.subplot(2, 2, 4)
        plt.hist(normalized_segment_lengths, bins=20)
        plt.axvline(self.results["avg_normalized_segment_length"], color='r', linestyle='dashed', linewidth=2)
        plt.title(f"Segment Length Distribution\nAvg: {self.results['avg_normalized_segment_length']:.3f}")

        plt.tight_layout()
        plt.show()

        print(f"Average normalized segment length: {self.results['avg_normalized_segment_length']:.3f}")
        print(f"Median normalized segment length: {self.results['median_normalized_segment_length']:.3f}")
        print(f"Number of segments: {self.results['segment_count']}")
        print(f"Number of contours: {self.results['total_contours']}")

    def compute(self, debug=False):
        """
        Run the complete analysis pipeline: read, process, analyze, and optionally visualize the results.
        """
        self.read_image()
        self.preprocess_image()
        self.find_contours()
        self.compute_contours()
        normalized_segment_lengths = self.calculate_statistics()

        if debug and self.segment_lengths:
            self.debug_visualization(normalized_segment_lengths)

        return self.results


if __name__ == "__main__":
    # Replace with your actual image file path
    image_path = "atest/5.png"
    analyzer = StrokeCurvatureContinuity(image_path)
    results = analyzer.compute(debug=True)
    print(results)
