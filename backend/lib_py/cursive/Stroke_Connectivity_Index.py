import base64
import os
import traceback
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np


class StrokeConnectivityAnalyzer:
    """
    Analyzes stroke connectivity characteristics of an image assumed to
    contain a single word, returning only numerical metrics.
    """
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the StrokeConnectivityAnalyzer.

        Args:
            image_input (str): Base64 encoded image string or image file path.
            is_base64 (bool): True if image_input is base64, False if file path.

        Raises:
            ValueError: If the image cannot be loaded, decoded, or is not grayscale/color.
            FileNotFoundError: If the image file path is invalid (and is_base64=False).
        """
        self.img = None
        self.gray_img = None
        self.input_type = "base64" if is_base64 else "filepath"
        self.avg_stroke_height = 10

        try:
            if is_base64:
                if not isinstance(image_input, str):
                    raise ValueError("Base64 input must be a string.")
                # Pad base64 string if needed
                missing_padding = len(image_input) % 4
                if missing_padding:
                    image_input += '=' * (4 - missing_padding)
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if self.img is None:
                    raise ValueError("Could not decode base64 image string.")
            else:
                if not isinstance(image_input, str):
                    raise ValueError("File path input must be a string.")
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found at {image_input}")
                self.img = cv2.imread(image_input)
                if self.img is None:
                    raise ValueError(
                        f"Could not read or decode image file at {image_input} (possibly corrupt or unsupported format)")

            # Convert to grayscale
            if len(self.img.shape) == 3 and self.img.shape[2] == 3:
                self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            elif len(self.img.shape) == 2:
                self.gray_img = self.img.copy()
            else:
                raise ValueError(f"Unsupported image format/shape: {self.img.shape}")

            if self.gray_img is None or self.gray_img.size == 0:
                raise ValueError("Grayscale conversion resulted in an empty image.")

            self.avg_stroke_height = self._estimate_stroke_height()

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise e
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise ValueError(f"Failed to initialize analyzer: {e}")

    def _estimate_stroke_height(self):
        """
        Estimates the average stroke height from the grayscale image using contour analysis
        on an inverted Otsu thresholded image.
        """
        if self.gray_img is None or self.gray_img.size == 0:
            return 10

        try:
            _, thresh = cv2.threshold(
                self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        except cv2.error as e:
            print(f"Warning: Otsu thresholding failed ({e}). Using simple threshold.")
            _, thresh = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heights = []
        min_area_for_height = 5
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area_for_height:
                try:
                    _, _, _, h = cv2.boundingRect(c)
                    if 1 < h < self.gray_img.shape[0] * 0.8:
                        heights.append(h)
                except cv2.error:
                    continue

        if not heights:
            return 10

        median_h = np.median(heights)
        estimated_height = max(5.0, min(median_h, 70.0))
        return estimated_height

    def _preprocess_image(self, adaptive_block_size=15, adaptive_c=5):
        """
        Preprocesses the grayscale image for analysis. Applies blur and adaptive thresholding.

        Args:
            adaptive_block_size (int): Block size for adaptive thresholding (must be odd and > 1).
            adaptive_c (int): Constant subtracted from the mean in adaptive thresholding.

        Returns:
            tuple: (binary image, bounding box tuple (x, y, w, h)) or (binary image, None) if no ink.
        """
        if self.gray_img is None:
            print("Error: Grayscale image is not available for preprocessing.")
            return None, None

        try:
            blurred = cv2.GaussianBlur(self.gray_img, (3, 3), 0)
            if adaptive_block_size % 2 == 0 or adaptive_block_size <= 1:
                adaptive_block_size = 15

            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_c
            )

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return binary, None

            x_coords, y_coords = [], []
            for cnt in contours:
                if cv2.contourArea(cnt) > 2:
                    x, y, w, h = cv2.boundingRect(cnt)
                    x_coords.extend([x, x + w])
                    y_coords.extend([y, y + h])

            if not x_coords or not y_coords:
                return binary, None

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            word_w = max(1, x_max - x_min)
            word_h = max(1, y_max - y_min)
            word_bbox = (x_min, y_min, word_w, word_h)
            return binary, word_bbox

        except cv2.error as e:
            print(f"OpenCV Error during preprocessing: {e}")
            return None, None
        except Exception as e:
            print(f"Unexpected Error during preprocessing: {e}")
            return None, None

    def _compute_metrics(self, binary_word_img, word_bbox):
        """
        Computes stroke connectivity metrics from the binarized single word image.

        Args:
            binary_word_img (numpy.ndarray): Binarized image (text=white).
            word_bbox (tuple): The (x, y, w, h) bounding box of the word ink.

        Returns:
            dict: Dictionary of computed numerical metrics.
        """
        default_metrics = {
            "total_components": 0,
            "total_ink_pixels": 0,
            "bounding_box_area": 0,
            "ink_density": 0.0,
            "component_density": 0.0,
            "average_component_area": 0.0,
            "median_component_area": 0.0,
            "component_area_std_dev": 0.0
        }

        if binary_word_img is None or binary_word_img.size == 0 or word_bbox is None:
            return default_metrics

        x, y, w, h = word_bbox
        if w <= 0 or h <= 0:
            return default_metrics

        try:
            word_img_cropped = binary_word_img[y:y + h, x:x + w]
            if word_img_cropped.size == 0:
                return default_metrics

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                word_img_cropped, connectivity=8
            )
            min_component_area = max(3, int((self.avg_stroke_height * 0.4) ** 2))

            valid_components_count = 0
            total_component_pixel_area = 0
            component_areas = []

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_component_area:
                    valid_components_count += 1
                    total_component_pixel_area += area
                    component_areas.append(area)

            if valid_components_count == 0:
                metrics = default_metrics.copy()
                metrics["bounding_box_area"] = w * h
                return metrics

            bounding_box_area = w * h
            actual_ink_pixels = total_component_pixel_area
            ink_density = actual_ink_pixels / bounding_box_area if bounding_box_area > 0 else 0.0
            component_density = (valid_components_count / bounding_box_area) * 1000 if bounding_box_area > 0 else 0.0
            average_component_area = np.mean(component_areas) if component_areas else 0.0
            median_component_area = np.median(component_areas) if component_areas else 0.0
            component_area_std_dev = np.std(component_areas) if component_areas else 0.0

            return {
                "total_components": valid_components_count,
                "total_ink_pixels": actual_ink_pixels,
                "bounding_box_area": bounding_box_area,
                "ink_density": ink_density,
                "component_density": component_density,
                "average_component_area": average_component_area,
                "median_component_area": median_component_area,
                "component_area_std_dev": component_area_std_dev
            }

        except cv2.error as e:
            print(f"OpenCV Error during metric computation: {e}")
            return default_metrics
        except Exception as e:
            print(f"Unexpected Error during metric computation: {e}")
            return default_metrics

    def analyze(self, debug=False, adaptive_block_size=15, adaptive_c=5):
        """
        Analyzes the single word image for stroke connectivity.

        Args:
            debug (bool): If True, generates and returns visualization plots.
            adaptive_block_size (int): Parameter for adaptive thresholding.
            adaptive_c (int): Parameter for adaptive thresholding.

        Returns:
            dict: Contains 'metrics' (numerical values only) and optionally 'graphs'
                  (if debug=True).
        """
        result = {'metrics': {}, 'graphs': [], 'preprocessed_image': None}
        default_metrics = self._compute_metrics(None, None)

        try:
            binary_img, word_bbox = self._preprocess_image(adaptive_block_size, adaptive_c)
            if binary_img is None or word_bbox is None:
                result['metrics'] = default_metrics
                return result

            metrics = self._compute_metrics(binary_img, word_bbox)
            result['metrics'] = metrics

            analysis_successful = metrics.get("total_components", 0) > 0 or metrics.get("bounding_box_area", 0) > 0

            if debug:
                try:
                    if len(self.img.shape) == 2:
                        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
                    elif len(self.img.shape) == 3 and self.img.shape[2] == 3:
                        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                    elif len(self.img.shape) == 3 and self.img.shape[2] == 4:
                        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGRA2RGB)
                    else:
                        raise ValueError("Cannot convert original image to RGB for plotting")
                except Exception as plot_img_err:
                    print(f"Warning: Could not prepare original image for plotting: {plot_img_err}")
                    img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

                try:
                    plt.style.use('seaborn-v0_8-darkgrid')
                except Exception:
                    pass

                fig = plt.figure("Single Word Connectivity Analysis", figsize=(10, 8))

                # Plot 1: Original Image with Bounding Box
                ax1 = plt.subplot(2, 2, 1)
                ax1.imshow(img_rgb)
                ax1.set_title("Original Image & BBox")
                ax1.axis('off')
                if word_bbox:
                    x, y, w, h = word_bbox
                    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=1)
                    ax1.add_patch(rect)

                # Plot 2: Binarized Image (Cropped)
                ax2 = plt.subplot(2, 2, 2)
                if word_bbox:
                    x, y, w, h = word_bbox
                    binary_cropped = binary_img[y:y + h, x:x + w]
                    if binary_cropped.size > 0:
                        ax2.imshow(binary_cropped, cmap='gray')
                        ax2.set_title("Binarized Word (Cropped)")
                    else:
                        ax2.text(0.5, 0.5, "Empty Crop", horizontalalignment='center', verticalalignment='center')
                        ax2.set_title("Binarized Word (Crop Failed)")
                else:
                    ax2.imshow(binary_img, cmap='gray')
                    ax2.set_title("Binarized Image (No BBox)")
                ax2.axis('off')

                # Plot 3: Components Visualization
                ax3 = plt.subplot(2, 2, 3)
                if analysis_successful and word_bbox:
                    try:
                        x, y, w, h = word_bbox
                        binary_cropped = binary_img[y:y + h, x:x + w]
                        if binary_cropped.size > 0:
                            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                                binary_cropped, connectivity=8
                            )
                            min_comp_area = max(3, int((self.avg_stroke_height * 0.4) ** 2))
                            labeled_img_bgr = np.zeros((binary_cropped.shape[0], binary_cropped.shape[1], 3),
                                                       dtype=np.uint8)
                            valid_component_count = 0
                            for i in range(1, num_labels):
                                if stats[i, cv2.CC_STAT_AREA] >= min_comp_area:
                                    valid_component_count += 1
                                    color = (np.random.randint(50, 256),
                                             np.random.randint(50, 256),
                                             np.random.randint(50, 256))
                                    labeled_img_bgr[labels == i] = color
                            labeled_img_rgb = cv2.cvtColor(labeled_img_bgr, cv2.COLOR_BGR2RGB)
                            ax3.imshow(labeled_img_rgb)
                            ax3.set_title(f"Filtered Components ({valid_component_count})")
                        else:
                            raise ValueError("Cropped binary image for component plot is empty.")
                    except Exception as comp_plot_err:
                        print(f"Warning: Could not generate component plot: {comp_plot_err}")
                        ax3.text(0.5, 0.5, "Plot Error", horizontalalignment='center', verticalalignment='center')
                        ax3.set_title("Filtered Components")
                else:
                    ax3.text(0.5, 0.5, "N/A" if word_bbox else "No BBox", horizontalalignment='center',
                             verticalalignment='center')
                    ax3.set_title("Filtered Components")
                ax3.axis('off')

                # Plot 4: Key Metrics Bar Chart
                ax4 = plt.subplot(2, 2, 4)
                if analysis_successful:
                    metric_names = ['Total\nComps', 'Comp Density\n(per 1k px)', 'Avg Comp\nArea (px)',
                                    'Median Comp\nArea (px)']
                    metric_values = [
                        metrics.get('total_components', 0),
                        metrics.get('component_density', 0),
                        metrics.get('average_component_area', 0),
                        metrics.get('median_component_area', 0)
                    ]
                    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
                    bars = ax4.bar(metric_names, metric_values, color=colors[:len(metric_values)])
                    ax4.bar_label(bars, fmt='{:.1f}', padding=3, fontsize=8)
                    ax4.set_title('Key Connectivity Metrics')
                    ax4.tick_params(axis='x', labelsize=8, rotation=10)
                    ax4.set_ylabel('Value', fontsize=9)
                    ax4.set_ylim(bottom=0)
                else:
                    reason = "No Components Found" if word_bbox else "Preprocessing Failed"
                    ax4.text(0.5, 0.5, f"Metrics N/A\n({reason})", horizontalalignment='center',
                             verticalalignment='center', fontsize=10)
                    ax4.set_title('Key Connectivity Metrics')
                ax4.set_xticks(range(len(metric_names)))
                ax4.set_xticklabels(metric_names)
                ax4.tick_params(axis='y', labelsize=8)

                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                try:
                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    result['graphs'].append(plot_base64)
                except Exception as save_plot_err:
                    print(f"Warning: Failed to save plot to base64: {save_plot_err}")
                finally:
                    plt.show()
                    plt.close(fig)

            try:
                _, buffer = cv2.imencode('.png', binary_img)
                preprocessed_image_base64 = base64.b64encode(buffer).decode('utf-8')
                result['preprocessed_image'] = preprocessed_image_base64
            except Exception as preprocess_encode_err:
                print(f"Warning: Failed to encode preprocessed image: {preprocess_encode_err}")
                result['preprocessed_image'] = None

        except Exception as e:
            print(f"Error during analysis execution: {e}")
            result['metrics'] = default_metrics
            result['graphs'] = []
            result['preprocessed_image'] = None

        if 'metrics' not in result or not result['metrics']:
            result['metrics'] = default_metrics

        return result


if __name__ == "__main__":
    image_path = '../../atest/1.png'
    ADAPTIVE_BLOCK_SIZE = 19  # Must be odd.
    ADAPTIVE_C = 6
    DEBUG_MODE = True

    print(f"\nAnalyzing single word image: {image_path}")
    print(f"Parameters: BlockSize={ADAPTIVE_BLOCK_SIZE}, C={ADAPTIVE_C}")
    print("-" * 30)

    try:
        analyzer = StrokeConnectivityAnalyzer(image_path, is_base64=False)
        results = analyzer.analyze(debug=DEBUG_MODE, adaptive_block_size=ADAPTIVE_BLOCK_SIZE, adaptive_c=ADAPTIVE_C)
        print("--- Analysis Results ---")
        metrics = results.get('metrics', {})
        if metrics.get('bounding_box_area', 0) > 0:
            print(f"Total Components Found: {metrics.get('total_components', 0)}")
            print(f"Bounding Box Area (pixels): {metrics.get('bounding_box_area', 0)}")
            print(f"Total Ink Pixels (filtered): {metrics.get('total_ink_pixels', 0)}")
            print(f"Ink Density (ink/bbox): {metrics.get('ink_density', 0):.3f}")
            print(f"Component Density (comps/1k px): {metrics.get('component_density', 0):.2f}")
            print(f"Average Component Area (pixels): {metrics.get('average_component_area', 0):.1f}")
            print(f"Median Component Area (pixels): {metrics.get('median_component_area', 0):.1f}")
            print(f"Component Area Std Dev: {metrics.get('component_area_std_dev', 0):.1f}")
            print(f"Preprocessed Image: {results.get('preprocessed_image')[:50] if results.get('preprocessed_image') else None}...")
        else:
            print("Analysis completed, but no word/ink detected (bounding box area is 0).")
            print(f"Metrics: {metrics}")
            print(f"Preprocessed Image: {results.get('preprocessed_image')[:50] if results.get('preprocessed_image') else None}...")
    except (ValueError, FileNotFoundError) as e:
        print("\n--- Analysis Failed (Setup Error) ---")
        print(f"Error: {e}")
    except Exception as e:
        print("\n--- Analysis Failed (Unexpected Error) ---")
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()