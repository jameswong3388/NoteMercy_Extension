import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO


class SlantAngleAnalyzer:
    def __init__(self, image_input, is_base64=False):
        """
        Initializes the SlantAngleAnalyzer with either a base64 encoded image or image path.

        Parameters:
            image_input (str): Either base64 encoded image string or image file path
            is_base64 (bool): If True, image_input is treated as base64 string, else as file path
        """
        self.contours = None
        self.slant_angles = []

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

    def _preprocess_image(self):
        """
        Preprocesses the image, thresholds it, and extracts significant contours.
        """
        _, binary = cv2.threshold(
            self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter out small contours
        self.contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    def _calculate_slant_angles(self):
        """
        Calculates the slant angles for each contour using moment analysis.
        
        Returns:
            list: List of slant angles in degrees
        """
        angles = []
        for cnt in self.contours:
            M = cv2.moments(cnt)
            # Skip contours with negligible area or near-zero second moment
            if M['m00'] == 0 or M['mu02'] < 1e-2:
                continue

            # Calculate skew using central moments
            skew = M['mu11'] / M['mu02']
            angle_rad = math.atan(skew)
            angle_deg = angle_rad * (180.0 / np.pi)
            angles.append(angle_deg)
        return angles

    def analyze(self, debug=False):
        """
        Analyzes the image to determine slant angle characteristics.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Dictionary containing metrics and graphs in base64 format
        """
        # Preprocess image and get contours
        self._preprocess_image()
        
        # Calculate slant angles
        self.slant_angles = self._calculate_slant_angles()
        
        # Compute metrics
        if self.slant_angles:
            avg_slant = np.mean(self.slant_angles)
            slant_std = np.std(self.slant_angles)
            # Convert to angle from vertical for intuitive interpretation
            vertical_slant = 90 - avg_slant if avg_slant <= 90 else avg_slant - 90
            
            metrics = {
                'avg_slant': avg_slant,
                'vertical_slant': vertical_slant,
                'slant_std': slant_std,
                'num_components': len(self.slant_angles)
            }
        else:
            metrics = {
                'avg_slant': 0,
                'vertical_slant': 0,
                'slant_std': 0,
                'num_components': 0
            }
        
        result = {
            'metrics': metrics,
            'graphs': []
        }
        
        # Generate visualization if debug mode is enabled
        if debug and self.slant_angles:
            # Create visualization
            vis_img = self.img.copy()
            cv2.drawContours(vis_img, self.contours, -1, (0, 255, 0), 2)
            
            # Create figure with subplots
            plt.figure("Slant Angle Analysis", figsize=(12, 6))
            
            # Original image with contours
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title("Contours")
            plt.axis('off')
            
            # Histogram of slant angles
            plt.subplot(122)
            plt.hist(self.slant_angles, bins=20, color='blue', alpha=0.7)
            plt.axvline(x=metrics['avg_slant'], color='r', linestyle='--',
                        label=f'Avg: {metrics["avg_slant"]:.1f}°, StdDev: {metrics["slant_std"]:.1f}°')
            plt.title("Slant Angle Distribution")
            plt.xlabel("Angle (degrees)")
            plt.ylabel("Frequency")
            plt.legend()
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
    image_path = "/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png"
    analyzer = SlantAngleAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])
