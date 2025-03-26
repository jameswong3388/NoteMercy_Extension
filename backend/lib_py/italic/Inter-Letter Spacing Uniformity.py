import cv2
import matplotlib.pyplot as plt
import numpy as np


class LetterSpacingAnalyzer:
    def __init__(self, image_path, debug=False):
        """
        Initializes the analyzer with the image path and debug flag.

        :param image_path: Path to the image file.
        :param debug: If True, displays debugging visualizations.
        """
        self.image_path = image_path
        self.debug = debug

    def analyze(self):
        """
        Reads the image, computes inter-letter spacing metrics, and returns a results dictionary.

        :return: Dictionary with spacing metrics and qualitative assessment.
        """
        # Read the image
        img = cv2.imread(self.image_path)
        if img is None:
            return {"error": "Image not found"}

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

        results = {}
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
            results = {
                "raw_avg_gap": float(avg_gap),
                "raw_gap_std": float(gap_std),
                "median_letter_width": float(median_width),
                "normalized_avg_gap": float(norm_avg_gap),
                "normalized_gap_std": float(norm_gap_std),
                "gap_count": len(gaps)
            }

            # Qualitative assessment
            if 0.1 <= norm_avg_gap <= 0.5 and norm_gap_std < 0.2:
                results["assessment"] = "Likely italic handwriting (moderate, consistent spacing)"
            elif norm_avg_gap < 0.1:
                results["assessment"] = "Likely cursive handwriting (minimal spacing)"
            elif norm_avg_gap > 0.5:
                results["assessment"] = "Likely printed handwriting (large spacing)"
            else:
                results["assessment"] = "Indeterminate style"

        # Debug visualization
        if self.debug and gaps:
            plt.figure(figsize=(10, 6))

            # Original image
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")

            # Binary image with contours
            plt.subplot(2, 2, 2)
            debug_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.title("Letter Detection")

            # Histogram of gaps
            plt.subplot(2, 2, 3)
            plt.hist(gaps, bins=20)
            plt.axvline(avg_gap, color='r', linestyle='dashed', linewidth=2)
            plt.title(f"Gap Distribution (Mean={avg_gap:.2f})")

            # Normalized gaps
            if median_width > 0:
                plt.subplot(2, 2, 4)
                norm_gaps = [g / median_width for g in gaps]
                plt.hist(norm_gaps, bins=20)
                plt.axvline(norm_avg_gap, color='r', linestyle='dashed', linewidth=2)
                plt.title(f"Normalized Gaps (Mean={norm_avg_gap:.2f})")

            plt.tight_layout()
            plt.show()

        return results


if __name__ == "__main__":
    # Replace with your actual image file path
    image_path = "atest/4.png"
    analyzer = LetterSpacingAnalyzer(image_path, debug=True)
    results = analyzer.analyze()
    print(results)
