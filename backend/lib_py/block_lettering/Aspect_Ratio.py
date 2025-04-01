import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class AspectRatioAnalyzer:
    """
    Analyzes properties of detected letter candidates in an image, primarily
    aspect ratio consistency. Includes a pre-check for continuous script
    to avoid processing connected handwriting styles with letter-based metrics.
    """

    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer by loading the image.
        """
        self.img = None
        if is_base64:
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Load as color
            if self.img is None:
                raise ValueError("Could not decode base64 image.")
        else:
            self.img = cv2.imread(image_input, cv2.IMREAD_COLOR)  # Load as color
            if self.img is None:
                raise ValueError(f"Could not read image at path: {image_input}")

        self.original_height, self.original_width = self.img.shape[:2]
        self.gray_image = None
        self.binary_image = None
        self.letter_contours = []
        self.letter_bboxes = []
        self.all_aspect_ratios = []
        self.is_likely_continuous = False

    def _reset_analysis_data(self):
        """
        Clears data from previous analysis runs.
        """
        self.gray_image = None
        self.binary_image = None
        self.letter_contours = []
        self.letter_bboxes = []
        self.all_aspect_ratios = []
        self.is_likely_continuous = False

    def _preprocess_image(self):
        """
        Applies preprocessing: Grayscale, Optional Blur, Adaptive Thresholding.
        Uses fixed internal constants for parameters.
        """
        # --- Fixed Preprocessing Parameters ---
        # Set blur kernel size (1 = no blur, 3, 5, etc. - must be odd if > 1)
        _BLUR_KSIZE = 3  # Kernel size for Gaussian Blur (must be odd > 1, or <=1 to disable)
        _THRESH_VALUE = 127  # Threshold value for cv2.threshold
        _THRESH_MAX_VALUE = 255  # Max value for thresholding
        _THRESH_TYPE = cv2.THRESH_BINARY_INV  # Invert: strokes become white

        # 1. Convert to Grayscale
        if len(self.img.shape) == 3:
            self.gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:  # Assume already grayscale
            self.gray_image = self.img.copy()

        processed = self.gray_image.copy()

        # 2. Noise Reduction (Optional)
        if _BLUR_KSIZE > 1:
            ksize = _BLUR_KSIZE if _BLUR_KSIZE % 2 != 0 else _BLUR_KSIZE + 1  # Ensure odd
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

        # 3. Thresholding
        ret, self.binary_image = cv2.threshold(processed, _THRESH_VALUE, _THRESH_MAX_VALUE, _THRESH_TYPE)

    def _check_for_continuity(self):
        """
        Analyzes initial contours to detect likely continuous script.
        Sets the self.is_likely_continuous flag.
        Uses fixed internal threshold.
        """
        # --- Continuity Check Parameter ---
        CONTINUITY_PERIMETER_RATIO_THRESHOLD = 0.3

        initial_contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not initial_contours:
            self.is_likely_continuous = False
            return

        perimeters = [cv2.arcLength(cnt, True) for cnt in initial_contours]
        total_perimeter = sum(perimeters)
        if total_perimeter == 0:
            self.is_likely_continuous = False
            return

        max_perimeter = max(perimeters)
        ratio = max_perimeter / total_perimeter
        # print(f"Continuity Check: Max perimeter ratio = {ratio:.3f} (Threshold = {CONTINUITY_PERIMETER_RATIO_THRESHOLD})")

        if ratio > CONTINUITY_PERIMETER_RATIO_THRESHOLD:
            self.is_likely_continuous = True
            # print("Continuity Check: Detected LIKELY CONTINUOUS script.")
        else:
            self.is_likely_continuous = False
            # print("Continuity Check: Detected LIKELY NON-CONTINUOUS script.")

    def _find_letter_candidates(self):
        """
        Finds and filters contours to isolate potential single letters.
        Uses fixed internal constants for filtering criteria (area, height,
        aspect ratio, perimeter) to exclude noise, dots, and continuous scripts.
        """
        # --- Fixed Filtering Parameters ---
        MIN_AREA = 40  # Min pixel area (filters noise, small dots)
        MAX_AREA_RATIO = 0.15  # Max area relative to image (filters large blobs)
        MIN_HEIGHT = 30  # Min pixel height (filters short noise/dots)
        ASPECT_RATIO_RANGE = (0.1, 4.0)  # Allowed Width/Height for a *letter*
        # Max contour length (pixels) - VERY sensitive to image size/resolution
        # Needs adjustment if image dimensions change significantly.
        MAX_PERIMETER = 1000

        # --- Check continuity flag first ---
        if self.is_likely_continuous:
            # print("Skipping letter candidate filtering due to likely continuous script.")
            self.letter_contours = []  # Ensure lists are empty
            self.letter_bboxes = []
            return  # Exit the method early

        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.letter_contours = []
        self.letter_bboxes = []
        max_abs_area = self.original_height * self.original_width * MAX_AREA_RATIO
        count_initial = len(contours)
        # Optional: Keep track of filtered counts for debugging if needed
        # count_filtered_... = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > max_abs_area:
                # count_filtered_area += 1
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h < MIN_HEIGHT:
                # count_filtered_height += 1
                continue

            aspect_ratio = w / float(h) if h > 0 else 0
            if not (ASPECT_RATIO_RANGE[0] <= aspect_ratio <= ASPECT_RATIO_RANGE[1]):
                # count_filtered_aspect += 1
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter > MAX_PERIMETER:
                # count_filtered_perimeter += 1
                continue

            # Passed all filters
            self.letter_contours.append(cnt)
            self.letter_bboxes.append((x, y, w, h))

        # print(f"Found {len(self.letter_bboxes)} candidates after filtering {count_initial} initial contours.")

    def _calculate_aspect_ratios(self):
        """
        Calculates aspect ratio for the found letter candidates.
        """
        if not self.letter_bboxes:
            # print("No valid letter candidates found to calculate aspect ratios.")
            self.all_aspect_ratios = []
            return

        self.all_aspect_ratios = [w / float(h) for (x, y, w, h) in self.letter_bboxes if h > 0]
        # Handle cases where h=0 if necessary, though MIN_HEIGHT filter should prevent this.
        # print(f"Calculated {len(self.all_aspect_ratios)} aspect ratios.")

    def _calculate_statistics(self):
        """
        Calculates summary statistics for the aspect ratios.
        """
        num_ratios = len(self.all_aspect_ratios)
        num_letters_found = len(self.letter_bboxes)  # Final count after filtering

        if num_ratios > 0:
            ratios_array = np.array(self.all_aspect_ratios)
            mean_ar = np.mean(ratios_array)
            median_ar = np.median(ratios_array)
            std_dev_ar = np.std(ratios_array)
            cv_ar = std_dev_ar / mean_ar if mean_ar != 0 else 0
            min_ar = np.min(ratios_array)
            max_ar = np.max(ratios_array)
        else:
            mean_ar, median_ar, std_dev_ar, cv_ar, min_ar, max_ar = 0, 0, 0, 0, 0, 0

        metrics = {
            'num_letter_candidates': num_letters_found,
            'num_aspect_ratios_calculated': num_ratios,
            'mean_aspect_ratio': mean_ar,
            'median_aspect_ratio': median_ar,
            'std_dev_aspect_ratio': std_dev_ar,
            'cv_aspect_ratio': cv_ar,
            'min_aspect_ratio': min_ar,
            'max_aspect_ratio': max_ar,
        }
        return metrics

    def _generate_visualization(self, metrics):
        """ Generates visualization plots. Parameter values are not shown in titles. """
        graphs = []
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.figure("Letter Aspect Ratio Analysis", figsize=(12, 10))  # Simplified title

        # Plot 1: Original
        plt.subplot(2, 2, 1);
        plt.imshow(img_rgb);
        plt.title("Original Image");
        plt.axis('off')

        # Plot 2: Preprocessed
        plt.subplot(2, 2, 2);
        if self.binary_image is not None:
            plt.imshow(self.binary_image, cmap='gray')
            plt.title("Preprocessed Image")  # Generic title
        else:
            plt.text(0.5, 0.5, "Preprocessing Failed", ha='center', va='center')
            plt.title("Preprocessed")
        plt.axis('off')

        # Plot 3: Detected Candidates
        plt.subplot(2, 2, 3)
        img_with_boxes = img_rgb.copy()
        num_candidates_found = metrics.get('num_letter_candidates', 0)
        if num_candidates_found > 0 and self.letter_bboxes:
            for (x, y, w, h) in self.letter_bboxes:
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
        plt.imshow(img_with_boxes)
        plt.title(f"Detected Letter Candidates ({num_candidates_found})")
        plt.axis('off')

        # Plot 4: Aspect Ratio Histogram
        plt.subplot(2, 2, 4)
        if metrics.get('num_aspect_ratios_calculated', 0) > 0 and self.all_aspect_ratios:
            num_bins = min(20, max(5, metrics['num_aspect_ratios_calculated'] // 2))
            plt.hist(self.all_aspect_ratios, bins=num_bins)
            plt.axvline(metrics['mean_aspect_ratio'], color='r', ls='--', lw=2,
                        label=f"Mean: {metrics['mean_aspect_ratio']:.2f}")
            plt.axvline(metrics['median_aspect_ratio'], color='g', ls=':', lw=2,
                        label=f"Median: {metrics['median_aspect_ratio']:.2f}")
            plt.legend()
            plt.title(
                f"Aspect Ratio Dist. (StdDev: {metrics['std_dev_aspect_ratio']:.3f}, CV: {metrics['cv_aspect_ratio']:.3f})")
            plt.xlabel("Aspect Ratio (Width / Height)")
            plt.ylabel("Frequency")
        else:
            plt.text(0.5, 0.5, "No suitable letter candidates found", ha='center', va='center',
                     transform=plt.gca().transAxes)
            plt.title("Aspect Ratio Distribution")

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        graphs.append(plot_base64)
        return graphs

    def analyze(self, debug=False):
        """
        Orchestrates the analysis using fixed internal parameters.
        Only 'debug' parameter controls optional visualization generation.

        Returns:
            dict: Contains 'metrics' dictionary and 'graphs' list (if debug=True).
        """
        # --- 0. Reset data ---
        self._reset_analysis_data()

        # --- 1. Preprocess ---
        self._preprocess_image()

        # *** KEY CHANGE: Call continuity check ***
        # --- 2. Check for Continuity ---
        self._check_for_continuity()

        # --- 3. Find & Filter Letter Candidates (now checks flag internally) ---
        self._find_letter_candidates()

        # --- 4. Calculate Aspect Ratios (only if letters found) ---
        self._calculate_aspect_ratios()

        # --- 5. Calculate Statistics ---
        metrics = self._calculate_statistics()  # Gets continuity flag too

        result = {
            'metrics': metrics,
            'graphs': [],
            'preprocessed_image': None,
        }

        # --- 6. Generate Visualization (Optional) ---
        if debug:
            result['graphs'] = self._generate_visualization(metrics=metrics)
            if self.binary_image is not None:
              _, preprocessed_buffer = cv2.imencode('.png', self.binary_image)
              result['preprocessed_image'] = base64.b64encode(preprocessed_buffer).decode('utf-8')

        # --- 7. Return Results ---
        return result


# === Example usage ===
if __name__ == "__main__":
    # --- Configuration ---
    image_path = r"C:\Users\Samson\Desktop\Coding\IPPR\NoteMercy_Extension\backend\atest\block-letters.jpg"  # <-- CHANGE THIS PATH if needed
    analyzer = AspectRatioAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    # print(results)

    # Print metrics in a readable format
    print("\n===== Aspect Ratio Analysis Results =====")
    metrics = results['metrics']
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Display the image directly without saving
    if results['graphs']:
        from PIL import Image
        import io

        print("\nDisplaying visualization...")
        img_data = base64.b64decode(results['graphs'][0])
        img = Image.open(io.BytesIO(img_data))
        img.show()
