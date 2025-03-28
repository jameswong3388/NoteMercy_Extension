import base64
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np


class SlantAngleAnalyzer:
    def __init__(self, image_input, is_base64=False):
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

        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_img = self.img.copy()

        self.contours = None
        self.slant_angles = []
        self.vertical_slants = []

    def _preprocess_image(self):
        _, binary = cv2.threshold(
            self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    def _calculate_slant_angles(self):
        angles = []
        for cnt in self.contours:
            if len(cnt) < 5:
                continue  # fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(cnt)
            angles.append(ellipse[2])  # Angle in degrees (0-180)
        return angles

    def analyze(self, debug=False, italic_threshold=8):
        self._preprocess_image()
        self.slant_angles = self._calculate_slant_angles()

        metrics = {}
        if self.slant_angles:
            self.vertical_slants = [90 - angle for angle in self.slant_angles]
            avg_vertical_slant = np.mean(self.vertical_slants)
            slant_std = np.std(self.vertical_slants)
            avg_slant = np.mean(self.slant_angles)
            is_italic = avg_vertical_slant > italic_threshold

            metrics = {
                'avg_slant': avg_slant,
                'vertical_slant': avg_vertical_slant,
                'slant_std': slant_std,
                'num_components': len(self.slant_angles),
                'is_italic': is_italic,
                'italic_threshold': italic_threshold
            }
        else:
            metrics = {
                'avg_slant': 0,
                'vertical_slant': 0,
                'slant_std': 0,
                'num_components': 0,
                'is_italic': False,
                'italic_threshold': italic_threshold
            }

        result = {'metrics': metrics, 'graphs': []}

        if debug and self.slant_angles:
            vis_img = self.img.copy()
            cv2.drawContours(vis_img, self.contours, -1, (0, 255, 0), 2)

            plt.figure("Slant Angle Analysis", figsize=(12, 6))

            plt.subplot(121)
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title("Contours")
            plt.axis('off')

            plt.subplot(122)
            plt.hist(self.vertical_slants, bins=20, color='blue', alpha=0.7)
            plt.axvline(
                x=metrics['vertical_slant'],
                color='r',
                linestyle='--',
                label=f'Avg: {metrics["vertical_slant"]:.1f}°, StdDev: {metrics["slant_std"]:.1f}°'
            )
            plt.axvline(x=italic_threshold, color='g', linestyle=':', label=f'Italic Threshold: {italic_threshold}°')
            plt.title("Vertical Slant Distribution (Right tilt positive)")
            plt.xlabel("Vertical Slant (degrees)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.show()
            plt.close()

            result['graphs'].append(plot_base64)

        return result


# Example usage
if __name__ == "__main__":
    image_path = "../../atest/print2.png"
    analyzer = SlantAngleAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True, italic_threshold=8)
    print(results['metrics'])