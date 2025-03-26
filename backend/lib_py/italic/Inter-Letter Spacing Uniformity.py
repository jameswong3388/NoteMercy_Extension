import cv2
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO


class LetterSpacingAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer with either a base64 encoded image or image path.

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

    def analyze(self, debug=False):
        """
        Reads the image, computes inter-letter spacing metrics, and returns results.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Metrics and visualization graphs (base64 encoded) if debug=True.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Binarize the image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours (letters)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get bounding boxes for each contour
        boxes = [cv2.boundingRect(cnt) for cnt in contours]

        # Sort contours by leftmost x coordinate
        boxes.sort(key=lambda b: b[0])

        # Calculate gaps between adjacent letters
        gaps = []
        for i in range(len(boxes) - 1):
            x_current_right = boxes[i][0] + boxes[i][2]
            x_next = boxes[i + 1][0]
            gap = x_next - x_current_right
            if gap > 0:  # Only consider positive gaps
                gaps.append(gap)

        metrics = {}
        if gaps:
            # Calculate spacing metrics
            avg_gap = np.mean(gaps)
            gap_std = np.std(gaps)

            # Normalize by median letter width for scale invariance
            median_width = np.median([w for x, y, w, h in boxes])
            if median_width > 0:
                norm_avg_gap = avg_gap / median_width
                norm_gap_std = gap_std / median_width
            else:
                norm_avg_gap = avg_gap
                norm_gap_std = gap_std

            # Store results
            metrics = {
                "raw_avg_gap": float(avg_gap),
                "raw_gap_std": float(gap_std),
                "median_letter_width": float(median_width),
                "normalized_avg_gap": float(norm_avg_gap),
                "normalized_gap_std": float(norm_gap_std),
                "gap_count": len(gaps)
            }

            # Qualitative assessment
            if 0.1 <= norm_avg_gap <= 0.5 and norm_gap_std < 0.2:
                metrics["assessment"] = "Likely italic handwriting (moderate, consistent spacing)"
            elif norm_avg_gap < 0.1:
                metrics["assessment"] = "Likely cursive handwriting (minimal spacing)"
            elif norm_avg_gap > 0.5:
                metrics["assessment"] = "Likely printed handwriting (large spacing)"
            else:
                metrics["assessment"] = "Indeterminate style"
        else:
            metrics = {
                "assessment": "Could not analyze letter spacing (no gaps detected)",
                "gap_count": 0
            }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # Debug visualization
        if debug and gaps:
            plt.figure(figsize=(10, 6))

            # Original image
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            # Binary image with contours
            plt.subplot(2, 2, 2)
            debug_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.title("Letter Detection")
            plt.axis('off')

            # Histogram of gaps
            plt.subplot(2, 2, 3)
            plt.hist(gaps, bins=20)
            plt.axvline(avg_gap, color='r', linestyle='dashed', linewidth=2)
            plt.title(f"Gap Distribution (Mean={avg_gap:.2f})")
            plt.xlabel("Gap Size (px)")
            plt.ylabel("Frequency")

            # Normalized gaps
            if median_width > 0:
                plt.subplot(2, 2, 4)
                norm_gaps = [g / median_width for g in gaps]
                plt.hist(norm_gaps, bins=20)
                plt.axvline(norm_avg_gap, color='r', linestyle='dashed', linewidth=2)
                plt.title(f"Normalized Gaps (Mean={norm_avg_gap:.2f})")
                plt.xlabel("Gap / Letter Width Ratio")
                plt.ylabel("Frequency")

            plt.tight_layout()
            
            # Convert plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            result['graphs'].append(plot_base64)

        return result


if __name__ == "__main__":
    # Replace with your actual image file path
    image_path = "atest/4.png"
    analyzer = LetterSpacingAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])
