import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


class SlantAngleAnalyzer:
    def __init__(self, image_path, debug=False):
        """
        Initialize the analyzer with the image path and debug mode.

        Args:
            image_path (str): Path to the image file.
            debug (bool): Flag to display debug visualizations.
        """
        self.image_path = image_path
        self.debug = debug
        self.img = None
        self.contours = None
        self.slant_angles = []
        self.results = {}

    def load_and_preprocess_image(self):
        """
        Loads the image, converts it to grayscale, thresholds it,
        and extracts significant contours.
        """
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise ValueError(f"Could not load image: {self.image_path}")

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter out small contours
        self.contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    def compute_slant_angle(self):
        """
        Computes the slant angles of contours in the image and calculates
        statistical values such as average slant, vertical slant from 90°,
        standard deviation, and number of components analyzed.

        Returns:
            dict: A dictionary containing the analysis results.
        """
        # Load and process the image if not already done
        if self.img is None or self.contours is None:
            self.load_and_preprocess_image()

        self.slant_angles = []
        for cnt in self.contours:
            M = cv2.moments(cnt)
            # Skip contours with negligible area or near-zero second moment
            if M['m00'] == 0 or M['mu02'] < 1e-2:
                continue

            # Calculate skew using central moments
            skew = M['mu11'] / M['mu02']
            angle_rad = math.atan(skew)
            angle_deg = angle_rad * (180.0 / np.pi)
            self.slant_angles.append(angle_deg)

        if self.slant_angles:
            avg_slant = np.mean(self.slant_angles)
            slant_std = np.std(self.slant_angles)
            # Convert to angle from vertical for intuitive interpretation
            vertical_slant = 90 - avg_slant if avg_slant <= 90 else avg_slant - 90

            self.results = {
                'avg_slant': avg_slant,
                'vertical_slant': vertical_slant,
                'slant_std': slant_std,
                'num_components': len(self.slant_angles)
            }
        else:
            self.results = {}

        if self.debug and self.slant_angles:
            self.debug_visualization()

        return self.results

    def debug_visualization(self):
        """
        Provides a debug visualization including the image with drawn contours
        and a histogram of the slant angles.
        """
        vis_img = self.img.copy()
        cv2.drawContours(vis_img, self.contours, -1, (0, 255, 0), 2)

        avg_slant = np.mean(self.slant_angles)
        slant_std = np.std(self.slant_angles)
        vertical_slant = 90 - avg_slant if avg_slant <= 90 else avg_slant - 90

        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title("Contours")

        plt.subplot(122)
        plt.hist(self.slant_angles, bins=20, color='blue', alpha=0.7)
        plt.axvline(x=avg_slant, color='r', linestyle='--',
                    label=f'Avg: {avg_slant:.1f}°, StdDev: {slant_std:.1f}°')
        plt.title("Slant Angle Distribution")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"Average slant: {avg_slant:.1f}°")
        print(f"Angle from vertical: {vertical_slant:.1f}°")
        print(f"Slant consistency (std): {slant_std:.1f}°")
        print(f"Components analyzed: {len(self.slant_angles)}")


if __name__ == "__main__":
    # Replace with your actual image file path
    image_path = "atest/3.png"
    analyzer = SlantAngleAnalyzer(image_path, debug=True)
    results = analyzer.compute_slant_angle()
    print(results)
