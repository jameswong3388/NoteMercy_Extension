import base64
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np


class LetterSpacingAnalyzer:
    def __init__(self, image_input, is_base64=True):
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

    def analyze(self, debug=False):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(cnt) for cnt in contours]
        boxes.sort(key=lambda b: b[0])

        gaps = []
        for i in range(len(boxes) - 1):
            x_current_right = boxes[i][0] + boxes[i][2]
            x_next = boxes[i + 1][0]
            gap = x_next - x_current_right
            if gap > 0:
                gaps.append(gap)

        metrics = {}
        if gaps:
            avg_gap = np.mean(gaps)
            gap_std = np.std(gaps)
            median_width = np.median([w for x, y, w, h in boxes])
            if median_width > 0:
                norm_avg_gap = avg_gap / median_width
                norm_gap_std = gap_std / median_width
            else:
                norm_avg_gap = avg_gap
                norm_gap_std = gap_std

            metrics = {
                "raw_avg_gap": float(avg_gap),
                "raw_gap_std": float(gap_std),
                "median_letter_width": float(median_width),
                "normalized_avg_gap": float(norm_avg_gap),
                "normalized_gap_std": float(norm_gap_std),
                "gap_count": len(gaps)
            }

            # Qualitative style assessment
            if 0.1 <= norm_avg_gap <= 0.5 and norm_gap_std < 0.2:
                metrics["assessment"] = "Likely italic handwriting (moderate spacing)"
            elif norm_avg_gap < 0.1:
                metrics["assessment"] = "Likely cursive handwriting (minimal spacing)"
            elif norm_avg_gap > 0.5:
                metrics["assessment"] = "Likely printed handwriting (large spacing)"
            else:
                metrics["assessment"] = "Indeterminate style"

            # Uniformity assessment
            if metrics['gap_count'] >= 2:
                if norm_avg_gap != 0:
                    cv = norm_gap_std / norm_avg_gap
                else:
                    cv = 0
                max_cv = 0.45  # Coefficient of variation threshold (15%)
                max_std = 0.4  # Normalized standard deviation threshold
                is_uniform = (cv <= max_cv) and (norm_gap_std <= max_std)
            else:
                is_uniform = None  # Not enough gaps to determine

            metrics['is_uniform'] = bool(is_uniform) if is_uniform is not None else None

            # Update assessment with uniformity
            if metrics['is_uniform'] is not None:
                if metrics['is_uniform']:
                    metrics["assessment"] += " with uniform spacing"
                else:
                    metrics["assessment"] += " with non-uniform spacing"
            elif metrics['gap_count'] == 1:
                metrics["assessment"] += "; cannot assess uniformity with one gap"
        else:
            metrics = {
                "assessment": "Could not analyze letter spacing (no gaps detected)",
                "gap_count": 0
            }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        if debug and gaps:
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(2, 2, 2)
            debug_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.title("Letter Detection")
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.hist(gaps, bins=20)
            plt.axvline(avg_gap, color='r', linestyle='dashed', linewidth=2)
            plt.title(f"Gap Distribution (Mean={avg_gap:.2f})")
            plt.xlabel("Gap Size (px)")
            plt.ylabel("Frequency")

            if median_width > 0:
                plt.subplot(2, 2, 4)
                norm_gaps = [g / median_width for g in gaps]
                plt.hist(norm_gaps, bins=20)
                plt.axvline(norm_avg_gap, color='r', linestyle='dashed', linewidth=2)
                plt.title(f"Normalized Gaps (Mean={norm_avg_gap:.2f})")
                plt.xlabel("Gap / Letter Width Ratio")
                plt.ylabel("Frequency")

            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            result['graphs'].append(plot_base64)

        # Preprocessed image base64
        _, preprocessed_image_encoded = cv2.imencode(".png", binary)
        preprocessed_image_base64 = base64.b64encode(preprocessed_image_encoded).decode('utf-8')

        result['preprocessed_image'] = preprocessed_image_base64

        return result


if __name__ == "__main__":
    image_path = "../../atest/print2.png"
    analyzer = LetterSpacingAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])