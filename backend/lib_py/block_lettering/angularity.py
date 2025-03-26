import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class BlockLetterAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the BlockLetterAnalyzer with either a base64 encoded image or image path.

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

    def _calculate_contour_area(self, contour):
        """Calculates the area of a given contour."""
        return cv2.contourArea(contour)

    def _measure_corner_angles(self, polygon):
        """
        Measures angular deviation at each vertex of the polygon.

        Parameters:
            polygon (np.ndarray): Array of vertices (x, y).

        Returns:
            list: List of deviation values from the nearest right angle (or 0, 90, 180, 270).
        """
        angles = []
        num_points = polygon.shape[0]
        for i in range(num_points):
            # Get three consecutive vertices with wrap-around
            p1 = polygon[i]
            p2 = polygon[(i + 1) % num_points]
            p3 = polygon[(i + 2) % num_points]

            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate the angle in degrees using arctan2 of determinant and dot product
            det = np.abs(np.linalg.det(np.array([v1, v2])))
            dot = np.dot(v1, v2)
            angle = np.degrees(np.arctan2(det, dot))

            # Compute deviation from the nearest target angle (0, 90, 180, 270)
            deviations = np.abs(angle - np.array([0, 90, 180, 270]))
            deviation = np.min(deviations)
            angles.append(deviation)
        return angles

    def _analyze_contour_geometry(self, contour):
        """
        Simplifies the contour and measures the corner angles.

        Parameters:
            contour (np.ndarray): Contour points.

        Returns:
            tuple: (simplified polygon, list of corner angles)
        """
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified_poly = approx.reshape(-1, 2)
        angles = self._measure_corner_angles(simplified_poly)
        return simplified_poly, angles

    def analyze(self, debug=False):
        """
        Analyzes the image to determine block letter characteristics by measuring
        angular deviations and corner features.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Metrics including average, median, maximum deviation,
                  number of corners, and number of letter shapes, plus visualization graphs if debug=True.
        """
        # Preprocess: Convert grayscale to binary image
        _, binary = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)  # Invert to make letters white

        # Find external contours (ignoring holes)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        corner_angles = []
        valid_contours = []

        # Process each contour and filter out small ones (noise)
        for cnt in contours:
            if self._calculate_contour_area(cnt) < 50:
                continue
            simplified_poly, angles = self._analyze_contour_geometry(cnt)
            corner_angles.extend(angles)
            valid_contours.append(cnt)

        # Compute block letter metrics
        if len(corner_angles) == 0:
            metrics = {
                'avg_deviation': 0,
                'median_deviation': 0,
                'max_deviation': 0,
                'corner_count': 0,
                'shape_count': len(valid_contours)
            }
        else:
            avg_deviation = np.mean(corner_angles)
            median_deviation = np.median(corner_angles)
            max_deviation = np.max(corner_angles)
            metrics = {
                'avg_deviation': avg_deviation,
                'median_deviation': median_deviation,
                'max_deviation': max_deviation,
                'corner_count': len(corner_angles),
                'shape_count': len(valid_contours)
            }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # Generate visualization plots if debug mode is enabled
        if debug:
            # Prepare a RGB version of the image for display
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            plt.figure("Block Letter Analysis", figsize=(10, 8))

            # Subplot 1: Original Image
            plt.subplot(2, 2, 1)
            plt.imshow(img_rgb)
            plt.title("Original Image")
            plt.axis('off')

            # Subplot 2: Binary Image (Extracted Letters)
            plt.subplot(2, 2, 2)
            plt.imshow(binary, cmap='gray')
            plt.title("Extracted Letters")
            plt.axis('off')

            # Subplot 3: Original Image with Detected Contours
            plt.subplot(2, 2, 3)
            plt.imshow(img_rgb)
            for cnt in valid_contours:
                cnt_squeezed = cnt.squeeze()
                if cnt_squeezed.ndim < 2:
                    continue
                closed = np.vstack([cnt_squeezed, cnt_squeezed[0]])
                plt.plot(closed[:, 0], closed[:, 1], 'r-', linewidth=2)
            plt.title("Corner Detection")
            plt.axis('off')

            # Subplot 4: Histogram of Angle Deviations
            plt.subplot(2, 2, 4)
            plt.hist(corner_angles, bins=20)
            plt.axvline(metrics['avg_deviation'], color='r', linestyle='--', linewidth=2)
            plt.title(f"Angle Distribution\nMean Deviation: {metrics['avg_deviation']:.2f}°")
            plt.xlabel("Deviation (°)")
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.show()
            
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
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png'
    analyzer = BlockLetterAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results)
