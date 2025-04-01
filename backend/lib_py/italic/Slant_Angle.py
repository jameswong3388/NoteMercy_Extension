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
        Generates a multi-panel figure visualizing the analysis including:
         - Original image with detected lines overlay.
         - Histogram of vertical slants.
         - Box plot of vertical slants.
         - A summary of computed metrics.

        Args:
            metrics (dict): The computed slant metrics.

        Returns:
            list: A list containing a base64-encoded PNG of the debug plot.
        """
        # Create a copy of the original image to draw the detected lines.
        vis_img = self.img.copy()
        if self.hough_lines is not None:
            for line in self.hough_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red lines

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Italic Detection Analysis', fontsize=16)

        # Plot 1: Original image with detected lines.
        axs[0, 0].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Detected Lines")
        axs[0, 0].axis('off')

        # Plot 2: Histogram of vertical slants.
        axs[0, 1].hist(self.vertical_slants, bins=20, color='skyblue', alpha=0.8, edgecolor='black')
        axs[0, 1].axvline(
            x=metrics['vertical_slant'],
            color='r',
            linestyle='--',
            linewidth=2,
            label=f'Avg: {metrics["vertical_slant"]:.1f}°'
        )
        axs[0, 1].axvline(
            x=metrics['italic_threshold'],
            color='g',
            linestyle=':',
            linewidth=2,
            label=f'Threshold: {metrics["italic_threshold"]}°'
        )
        axs[0, 1].set_title("Vertical Slant Distribution")
        axs[0, 1].set_xlabel("Deviation from Vertical (°)")
        axs[0, 1].set_ylabel("Frequency")
        axs[0, 1].legend()
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.6)

        # Plot 3: Box Plot of vertical slants.
        axs[1, 0].boxplot(self.vertical_slants, vert=False, showmeans=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue'),
                          medianprops=dict(color='red', linewidth=2),
                          meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='red'))
        axs[1, 0].axvline(
            x=metrics['italic_threshold'],
            color='g',
            linestyle=':',
            linewidth=2,
            label=f'Threshold: {metrics["italic_threshold"]}°'
        )
        axs[1, 0].set_title("Vertical Slant Box Plot")
        axs[1, 0].set_xlabel("Deviation from Vertical (°)")
        axs[1, 0].set_yticks([])
        axs[1, 0].legend(loc='lower right')
        axs[1, 0].grid(axis='x', linestyle='--', alpha=0.6)

        # Plot 4: Metrics Summary.
        axs[1, 1].axis('off')
        axs[1, 1].set_title("Analysis Metrics", pad=20)
        metrics_text = (
            f"Avg. Vertical Slant: {metrics['vertical_slant']:.2f}°\n"
            f"Std Dev: {metrics['slant_std']:.2f}°\n"
            f"Total Lines: {metrics['num_lines']}\n"
            f"Italic Threshold: {metrics['italic_threshold']}°\n"
            f"Italic Detected: {'Yes' if metrics['is_italic'] else 'No'}"
        )
        axs[1, 1].text(0.5, 0.5, metrics_text,
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return [plot_base64]

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