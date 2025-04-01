# We might need scipy for more robust peak finding, but let's start simple
# from scipy.signal import find_peaks

import base64
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np


class LetterSpacingAnalyzer:
    """
    Analyzes the internal spacing characteristics of a single word image
    using its Vertical Projection Profile (VPP).

    Assumes the input image contains primarily one word.
    """
    def __init__(self, image_input, is_base64=True):
        if is_base64:
            try:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if self.img is None:
                    raise ValueError("Error: Could not decode base64 image")
            except Exception as e:
                raise ValueError(f"Error decoding base64 image: {e}")
        else:
            self.img = cv2.imread(image_input)
            if self.img is None:
                raise ValueError(f"Error: Could not read image at {image_input}")

    def preprocess_image(self):
        """
        Converts the image to grayscale, applies blurring, adaptive thresholding,
        and morphological closing to get a clean binary representation of the word.
        Returns the binary image (white text on black background).
        """
        if self.img is None:
            raise ValueError("Image not loaded correctly.")
        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        elif len(self.img.shape) == 2:
            gray = self.img  # Already grayscale
        else:
            raise ValueError(f"Unsupported image format with shape: {self.img.shape}")

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack([cnt for cnt in contours])
            x, y, w, h = cv2.boundingRect(all_points)
            padding = 2
            y1 = max(0, y - padding)
            y2 = min(closed.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(closed.shape[1], x + w + padding)
            cropped = closed[y1:y2, x1:x2]
            return cropped
        else:
            return closed

    def calculate_vpp(self, binary_image):
        """
        Calculates the Vertical Projection Profile (sum of white pixels per column).
        """
        if binary_image is None or binary_image.size == 0:
            return np.array([])
        vpp = np.sum(binary_image // 255, axis=0)
        return vpp

    def analyze_vpp(self, vpp):
        """
        Analyzes the VPP to find valleys (potential gaps) and estimate spacing.
        Returns:
            dict: Containing spacing metrics like average gap width, std dev, etc.
                  Returns default values if VPP is too short or flat.
        """
        if len(vpp) < 3:
            return {
                "avg_valley_width": 0.0, "valley_width_std": 0.0,
                "avg_peak_dist": 0.0, "peak_dist_std": 0.0,
                "valley_count": 0, "peak_count": 0,
                "relative_spacing_metric": 0.0
            }

        valleys_idx = []
        valley_widths = []
        current_valley_start = -1

        gap_threshold = 1

        for i in range(len(vpp)):
            is_valley_col = vpp[i] <= gap_threshold
            if is_valley_col and current_valley_start == -1:
                current_valley_start = i
            elif not is_valley_col and current_valley_start != -1:
                valley_width = i - current_valley_start
                if valley_width > 0:
                    valley_widths.append(valley_width)
                    valleys_idx.append(current_valley_start + valley_width // 2)
                current_valley_start = -1

        if current_valley_start != -1:
            valley_width = len(vpp) - current_valley_start
            if valley_width > 0:
                valley_widths.append(valley_width)
                valleys_idx.append(current_valley_start + valley_width // 2)

        peaks_idx = []
        min_peak_height = max(2, 0.15 * np.max(vpp)) if np.max(vpp) > 0 else 2
        for i in range(1, len(vpp) - 1):
            if vpp[i] > vpp[i - 1] and vpp[i] > vpp[i + 1] and vpp[i] >= min_peak_height:
                peaks_idx.append(i)

        avg_valley_width = float(np.mean(valley_widths)) if valley_widths else 0.0
        valley_width_std = float(np.std(valley_widths)) if len(valley_widths) > 1 else 0.0

        peak_distances = np.diff(peaks_idx) if len(peaks_idx) > 1 else []
        avg_peak_dist = float(np.mean(peak_distances)) if len(peak_distances) > 0 else 0.0
        peak_dist_std = float(np.std(peak_distances)) if len(peak_distances) > 1 else 0.0

        relative_spacing_metric = avg_valley_width / avg_peak_dist if avg_peak_dist > 0 else 0.0

        return {
            "avg_valley_width": avg_valley_width,
            "valley_width_std": valley_width_std,
            "avg_peak_dist": avg_peak_dist,
            "peak_dist_std": peak_dist_std,
            "valley_count": len(valley_widths),
            "peak_count": len(peaks_idx),
            "relative_spacing_metric": relative_spacing_metric,
        }

    def analyze(self, debug=False):
        try:
            binary = self.preprocess_image()
            if binary is None or binary.size == 0:
                return {
                    "metrics": {
                        "avg_valley_width": 0.0, "valley_width_std": 0.0,
                        "avg_peak_dist": 0.0, "peak_dist_std": 0.0,
                        "valley_count": 0, "peak_count": 0,
                        "relative_spacing_metric": 0.0,
                    },
                    "graphs": [],
                    "preprocessed_image": ""
                }

            vpp = self.calculate_vpp(binary)
            vpp_metrics = self.analyze_vpp(vpp)

            metrics = {**vpp_metrics}

            result = {"metrics": metrics, "graphs": []}

            if debug:
                plt.figure(figsize=(10, 10))

                plt.subplot(3, 1, 1)
                if self.img is not None:
                    plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
                    plt.title("Original Image")
                    plt.axis("off")
                else:
                    plt.text(0.5, 0.5, "Original image load error", ha='center', va='center')
                    plt.axis("off")

                plt.subplot(3, 1, 2)
                if binary is not None and binary.size > 0:
                    plt.imshow(binary, cmap='gray')
                    plt.title("Preprocessed Binary Word")
                    plt.axis("off")
                else:
                    plt.text(0.5, 0.5, "Preprocessing failed", ha='center', va='center')
                    plt.axis("off")

                plt.subplot(3, 1, 3)
                if len(vpp) > 0:
                    plt.plot(vpp, label='VPP')
                    peaks_idx = []
                    min_peak_height = max(2, 0.15 * np.max(vpp)) if np.max(vpp) > 0 else 2
                    for i in range(1, len(vpp) - 1):
                        if vpp[i] > vpp[i - 1] and vpp[i] > vpp[i + 1] and vpp[i] >= min_peak_height:
                            peaks_idx.append(i)
                    if peaks_idx:
                        plt.plot(peaks_idx, vpp[peaks_idx], "x", color='red', label='Detected Peaks')

                    plt.title(
                        f"Vertical Projection Profile (Peaks: {vpp_metrics['peak_count']}, Valleys: {vpp_metrics['valley_count']})")
                    plt.xlabel("Column Index")
                    plt.ylabel("Sum of White Pixels")
                    plt.legend()
                    plt.grid(True)
                else:
                    plt.text(0.5, 0.5, "VPP could not be calculated", ha='center', va='center')
                    plt.axis("off")

                plt.tight_layout()
                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                result["graphs"].append(plot_base64)
                plt.close()

            preprocessed_base64 = ""
            if binary is not None and binary.size > 0:
                _, preprocessed_encoded = cv2.imencode(".png", binary)
                preprocessed_base64 = base64.b64encode(preprocessed_encoded).decode("utf-8")
            result["preprocessed_image"] = preprocessed_base64

            return result

        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            return {
                "metrics": {},
                "graphs": [],
                "preprocessed_image": ""
            }


# Example Usage (assuming you have a word image)
if __name__ == "__main__":
    image_path = "../../atest/3.png"
    analyzer = LetterSpacingAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])