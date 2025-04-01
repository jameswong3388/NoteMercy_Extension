import base64
from io import BytesIO

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use a non-interactive backend suitable for saving figures
matplotlib.use('Agg')


class SlantAngleAnalyzer:
    def __init__(self, image_input, is_base64=False):
        """
        Initializes the detector with an image.

        Args:
            image_input (str): Path to the image file or a base64 encoded string.
            is_base64 (bool): True if image_input is a base64 string; False if it's a file path.
        """
        try:
            if is_base64:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if self.img is None:
                    raise ValueError("Error: Could not decode base64 image")
            else:
                self.img = cv2.imread(image_input)
                if self.img is None:
                    raise ValueError(f"Error: Could not read image at {image_input}")

            # Ensure we have a grayscale image for processing.
            if len(self.img.shape) == 3 and self.img.shape[2] == 3:
                self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            elif len(self.img.shape) == 2:
                self.gray_img = self.img.copy()
                self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f"Error: Unexpected image shape {self.img.shape}")
        except Exception as e:
            print(f"Error during image loading/preprocessing: {e}")
            raise

        self.edges = None
        self.hough_lines = None
        self.vertical_slants = []

    def _preprocess_image(self):
        """
        Converts the image to a binary format using Otsu's thresholding.
        """
        # Apply Otsu's thresholding on the grayscale image and invert so text is white.
        _, self.binary_img = cv2.threshold(
            self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def _detect_edges(self):
        """
        Uses Canny edge detection to extract edges from the binary image.
        """
        # Use Canny edge detector. Parameters can be tuned based on image quality.
        self.edges = cv2.Canny(self.binary_img, 50, 150)

    def _detect_lines(self):
        """
        Detects line segments using the probabilistic Hough Transform.
        """
        # Adjust these parameters based on image scale.
        min_line_length = max(20, self.img.shape[1] // 10)  # e.g., at least 10% of image width
        max_line_gap = 5

        # HoughLinesP returns lines in the form [x1, y1, x2, y2]
        self.hough_lines = cv2.HoughLinesP(self.edges,
                                           rho=1,
                                           theta=np.pi / 180,
                                           threshold=50,
                                           minLineLength=min_line_length,
                                           maxLineGap=max_line_gap)

    def _calculate_slant_angles(self):
        """
        Calculates the vertical slant (deviation from 90°) for each detected near-vertical line.

        Returns:
            list: A list of vertical slant angles (positive for rightward, negative for leftward).
        """
        vertical_slants = []
        if self.hough_lines is None:
            return vertical_slants

        for line in self.hough_lines:
            x1, y1, x2, y2 = line[0]
            # Update: Ensure consistent ordering by making the top point first.
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            # Calculate angle (in degrees) of the line segment relative to the horizontal axis.
            angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            if angle < 0:
                angle += 180

            # Focus on near-vertical lines (e.g., between 60° and 120°)
            if 60 <= angle <= 120:
                # Vertical slant: positive means a rightward tilt.
                v_slant = 90 - angle
                vertical_slants.append(v_slant)

        self.vertical_slants = vertical_slants
        return vertical_slants

    def _generate_debug_plots(self, metrics):
        """
        Generates four separate figures for a detailed analysis:
          1. Detected Lines overlay on the original image.
          2. Histogram of vertical slants with mean, median, and threshold lines.
          3. Box plot of vertical slants with a threshold marker.
          4. Cumulative distribution function (CDF) of vertical slants.

        Args:
            metrics (dict): The computed slant metrics.

        Returns:
            list: A list of base64 encoded PNG images for each graph.
        """
        graphs_base64 = []

        # Graph 1: Detected Lines with Hough Transform Overlay
        vis_img = self.img.copy()
        if self.hough_lines is not None:
            for line in self.hough_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        fig1 = plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Lines (Total: {metrics['num_lines']})")
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graphs_base64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        plt.close(fig1)

        # Graph 2: Histogram of Vertical Slants with Annotations
        fig2 = plt.figure(figsize=(8, 6))
        bins = 20
        plt.hist(self.vertical_slants, bins=bins, color='skyblue', alpha=0.8, edgecolor='black')
        mean_slant = np.mean(self.vertical_slants) if self.vertical_slants else 0
        median_slant = np.median(self.vertical_slants) if self.vertical_slants else 0
        plt.axvline(x=mean_slant, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_slant:.1f}°')
        plt.axvline(x=median_slant, color='b', linestyle='-', linewidth=2, label=f'Median: {median_slant:.1f}°')
        plt.axvline(x=metrics['italic_threshold'], color='g', linestyle=':', linewidth=2,
                    label=f'Threshold: {metrics["italic_threshold"]}°')
        plt.xlabel("Deviation from Vertical (°)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Vertical Slants")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graphs_base64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        plt.close(fig2)

        # Graph 3: Box Plot of Vertical Slants with Threshold Annotation
        fig3 = plt.figure(figsize=(8, 4))
        plt.boxplot(self.vertical_slants, vert=False, showmeans=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='red'))
        plt.axvline(x=metrics['italic_threshold'], color='g', linestyle=':', linewidth=2,
                    label=f'Threshold: {metrics["italic_threshold"]}°')
        plt.xlabel("Deviation from Vertical (°)")
        plt.title("Box Plot of Vertical Slants")
        plt.legend(loc='lower right')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graphs_base64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        plt.close(fig3)

        # Graph 4: Cumulative Distribution Function (CDF) of Vertical Slants
        fig4 = plt.figure(figsize=(8, 6))
        if self.vertical_slants:
            sorted_slants = np.sort(self.vertical_slants)
            cdf = np.arange(1, len(sorted_slants) + 1) / len(sorted_slants)
            plt.plot(sorted_slants, cdf, marker='o', linestyle='-', color='purple')
            plt.axvline(x=mean_slant, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_slant:.1f}°')
            plt.axvline(x=metrics['italic_threshold'], color='g', linestyle=':', linewidth=2,
                        label=f'Threshold: {metrics["italic_threshold"]}°')
        else:
            plt.text(0.5, 0.5, "No slant data available", horizontalalignment='center', verticalalignment='center')
        plt.xlabel("Deviation from Vertical (°)")
        plt.ylabel("Cumulative Proportion")
        plt.title("Cumulative Distribution of Vertical Slants")
        plt.legend()
        plt.grid(linestyle='--', alpha=0.6)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graphs_base64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        plt.close(fig4)

        return graphs_base64

    def _encode_image_to_base64(self, image):
        """
        Encodes a given image to base64.
        """
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze(self, debug=False, italic_threshold=8):
        """
        Performs italic detection by computing the average vertical slant from detected line segments.

        Args:
            debug (bool): If True, generates and returns visualization plots.
            italic_threshold (float): The threshold (in degrees) for the average vertical slant above which
                                      the text is considered italic.

        Returns:
            dict: Contains:
                  'metrics': A dictionary of computed slant metrics.
                  'graphs': A list of base64 encoded PNG strings of the debug plots (if debug=True).
                  'preprocessed_image': Base64 encoded version of the binary (thresholded) image.
        """
        self._preprocess_image()
        self._detect_edges()
        self._detect_lines()
        slants = self._calculate_slant_angles()

        # Compute metrics only if we have valid slant values.
        if slants:
            avg_vertical_slant = float(np.mean(slants))
            slant_std = float(np.std(slants))
            is_italic = abs(avg_vertical_slant) > italic_threshold  # positive indicates rightward slant
            num_lines = len(slants)
        else:
            avg_vertical_slant = 0.0
            slant_std = 0.0
            is_italic = False
            num_lines = 0

        metrics = {
            'vertical_slant': avg_vertical_slant,
            'slant_std': slant_std,
            'num_lines': num_lines,
            'is_italic': is_italic,
            'italic_threshold': float(italic_threshold)
        }

        result = {'metrics': metrics, 'graphs': []}
        if debug:
            result['graphs'] = self._generate_debug_plots(metrics)

        # Return the binary (thresholded) image for reference.
        result['preprocessed_image'] = self._encode_image_to_base64(cv2.cvtColor(self.binary_img, cv2.COLOR_GRAY2BGR))
        return result

if __name__ == "__main__":
    image_path = "../../atest/shorthand2.png"
    italic_detector = SlantAngleAnalyzer(image_path)
    results = italic_detector.analyze(debug=True)
    print(results['metrics'])