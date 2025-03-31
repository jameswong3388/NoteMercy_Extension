import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class ContinuousPartCoverageAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer by loading the original color image.
        """
        self.img_color = None

        if is_base64:
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            self.img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if self.img_color is None:
                raise ValueError("Could not decode base64 image as color.")
        else:
            self.img_color = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if self.img_color is None:
                raise ValueError(f"Could not read image as color at path: {image_input}")

        self.original_height, self.original_width = self.img_color.shape[:2]
        self.binary_image = None
        self.contours = []
        self.total_bounding_box = None
        self.continuous_part_boxes = []

    def _reset_analysis_data(self):
        """
        Clears intermediate data from previous analysis runs.
        """
        self.binary_image = None
        self.contours = []
        self.total_bounding_box = None
        self.continuous_part_boxes = []

    def preprocess_image(self):
        """
        Applies preprocessing: Grayscale, Blur, Otsu Thresholding.
        Stores results in self.binary_image.
        Uses fixed internal parameters.
        """
        BLUR_KSIZE = 7

        # 1. Grayscale
        gray_image = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        processed = gray_image.copy()

        # 2. Gaussian Blur
        if BLUR_KSIZE > 1:
            ksize = BLUR_KSIZE if BLUR_KSIZE % 2 != 0 else BLUR_KSIZE + 1
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

        # 3. Adaptive Thresholding
        self.binary_image = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # # 4. Morphological Closing
        # kernel = np.ones((3, 3), np.uint8)
        # self.binary_image = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    def _calculate_coverage_ratio(self):
        """
        Calculates the Continuous Part Coverage Ratio using fixed internal parameters.
        Corrects for overlaps.
        """
        CONTINUOUS_PART_THRESHOLD_FACTOR = 0.1775

        metrics = {
            'total_text_width': 0,
            'continuous_part_count': 0,
            'sum_individual_part_widths': 0,
            'effective_covered_width': 0,
            'continuous_part_coverage_ratio': 0.0,
            'threshold_factor_used': CONTINUOUS_PART_THRESHOLD_FACTOR
        }

        plot_data = {
            'total_bounding_box': None,
            'continuous_part_boxes': []
        }

        self.contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not self.contours:
            return metrics, plot_data

        all_points = np.vstack(self.contours).squeeze()

        if all_points.ndim == 1:
            all_points = np.array([all_points])

        if all_points.size == 0:
            return metrics, plot_data

        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        total_width = max(1, max_x - min_x + 1)
        total_height = max(1, max_y - min_y + 1)
        self.total_bounding_box = (min_x, min_y, total_width, total_height)
        metrics['total_text_width'] = total_width
        plot_data['total_bounding_box'] = self.total_bounding_box
        width_threshold_pixels = total_width * CONTINUOUS_PART_THRESHOLD_FACTOR
        self.continuous_part_boxes = []
        sum_individual_widths = 0

        for cnt in self.contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > width_threshold_pixels:
                self.continuous_part_boxes.append((x, y, w, h))
                sum_individual_widths += w

        plot_data['continuous_part_boxes'] = self.continuous_part_boxes
        metrics['continuous_part_count'] = len(self.continuous_part_boxes)
        metrics['sum_individual_part_widths'] = sum_individual_widths

        if not self.continuous_part_boxes:
            metrics['effective_covered_width'] = 0
        else:
            horizontal_coverage_mask = np.zeros(total_width, dtype=bool)
            for (x_part, _, w_part, _) in self.continuous_part_boxes:
                mask_start_idx = max(0, x_part - min_x)
                mask_end_idx = min(total_width, x_part + w_part - min_x)

                if mask_end_idx > mask_start_idx:
                     horizontal_coverage_mask[mask_start_idx:mask_end_idx] = True

            effective_width = np.sum(horizontal_coverage_mask)
            metrics['effective_covered_width'] = effective_width

        if total_width > 0:
             metrics['continuous_part_coverage_ratio'] = metrics['effective_covered_width'] / total_width
        else:
             metrics['continuous_part_coverage_ratio'] = 0.0

        return metrics, plot_data

    def _generate_visualization(self, metrics, plot_data):
        """
        Generates visualization plots, including a coverage ratio bar chart.
        Uses fixed display parameters.
        """
        # --- Fixed Visualization Parameters ---
        FIGURE_SIZE = (12, 8)
        TOTAL_BOX_COLOR = (0, 0, 255) # Blue in BGR
        PART_BOX_COLOR = (0, 255, 0) # Green in BGR
        BOX_THICKNESS = 2
        # Colors for the bar chart segments
        COVERED_BAR_COLOR = 'seagreen'
        UNCOVERED_BAR_COLOR = 'lightcoral'
        LAYOUT_PADDING = 1.5

        graphs = []
        if self.img_color is None or self.binary_image is None or plot_data is None:
            print("Warning: Cannot generate visualization, required data missing.")
            return graphs

        plt.figure("Continuous Part Coverage Analysis", figsize=FIGURE_SIZE)

        # Plot 1: Original Image
        plt.subplot(2, 2, 1)
        img_rgb = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis('off')

        # Plot 2: Preprocessed Image
        plt.subplot(2, 2, 2)
        plt.imshow(self.binary_image, cmap='gray')
        plt.title("Preprocessed Image")
        plt.axis('off')

        # Plot 3: Detected Continuous Parts
        plt.subplot(2, 2, 3)
        vis_img_boxes = img_rgb.copy()
        if plot_data.get('total_bounding_box'):
            x_t, y_t, w_t, h_t = plot_data['total_bounding_box']
            cv2.rectangle(vis_img_boxes, (x_t, y_t), (x_t + w_t, y_t + h_t), TOTAL_BOX_COLOR, BOX_THICKNESS)
        cont_part_boxes = plot_data.get('continuous_part_boxes', [])
        for (x, y, w, h) in cont_part_boxes:
            cv2.rectangle(vis_img_boxes, (x, y), (x + w, y + h), PART_BOX_COLOR, BOX_THICKNESS)
        plt.imshow(vis_img_boxes)
        plt.title(f"Total Box (Blue), Continuous Parts (Green, {len(cont_part_boxes)})")
        plt.axis('off')

        # --- Plot 4: Coverage Ratio Stacked Bar Chart ---
        plt.subplot(2, 2, 4)
        total_w = metrics.get('total_text_width', 0)
        effective_w = metrics.get('effective_covered_width', 0)
        ratio = metrics.get('continuous_part_coverage_ratio', 0.0)

        # Ensure effective width doesn't exceed total width (can happen with rounding)
        effective_w = min(effective_w, total_w)
        uncovered_w = total_w - effective_w

        # Data for the stacked bar chart
        widths = [effective_w, uncovered_w]
        labels = ['Covered by Parts', 'Uncovered']
        colors = [COVERED_BAR_COLOR, UNCOVERED_BAR_COLOR]
        y_pos = 0 # Position on y-axis (only one bar)

        # Plot the bars stacked horizontally
        plt.barh(y_pos, widths[0], color=colors[0], label=labels[0])
        plt.barh(y_pos, widths[1], left=widths[0], color=colors[1], label=labels[1])

        # Add labels and title
        plt.xlabel("Width (pixels)")
        plt.title(f"Continuous Part Coverage Ratio: {ratio:.3f}")
        plt.yticks([], []) # Hide y-axis ticks as it's just one category
        plt.legend(loc='lower right', fontsize='small')

        # Set x-axis limits to show the full bar nicely
        plt.xlim(0, total_w * 1.05) # Add a little padding

        # Optional: Add text annotation for the ratio value
        plt.text(total_w / 2, y_pos, f"{ratio*100:.1f}%",
                 ha='center', va='center', color='white', fontweight='bold')


        plt.tight_layout(pad=LAYOUT_PADDING)

        # Save plot to buffer and encode
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close() # Close the figure explicitly

        graphs.append(plot_base64)
        return graphs

    def analyze(self, debug=False):
        """
        Orchestrates the analysis process with fixed internal parameters.
        """
        self._reset_analysis_data()
        self.preprocess_image()
        metrics, plot_data = self._calculate_coverage_ratio()

        result = {'metrics': metrics, 'graphs': []}

        if 'error' in metrics:
            return result

        if debug:
            if plot_data and plot_data.get('total_bounding_box') is not None:
                result['graphs'] = self._generate_visualization(metrics, plot_data)
            else:
                print("Warning: Skipping visualization due to missing plot data.")

        # Preprocess the image and convert to base64
        _, buffer = cv2.imencode('.png', self.binary_image)
        preprocessed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        result['preprocessed_image'] = preprocessed_image_base64

        return result


# === Example Usage (remains the same) ===
if __name__ == "__main__":
    image_path = r"C:\Users\Samson\Desktop\Coding\IPPR\NoteMercy_Extension\backend\atest\calligraphic.png" # <<< CHANGE THIS
    analyzer = ContinuousPartCoverageAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)

    print("\n===== Continuous Part Coverage Analysis Results =====")
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
    print("\nDisplaying preprocessed_image...")
    preprocessed_img_data = base64.b64decode(results['preprocessed_image'])
    preprocessed_img = Image.open(io.BytesIO(preprocessed_img_data))
    preprocessed_img.show()