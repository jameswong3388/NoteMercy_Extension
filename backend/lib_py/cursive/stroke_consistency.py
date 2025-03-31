import cv2
import numpy as np
import base64
from skimage.morphology import skeletonize


class StrokeConsistencyAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initialize the analyzer with either a base64 encoded image or image path.

        Parameters:
            image_input (str): Either base64 encoded image string or image file path
            is_base64 (bool): If True, image_input is treated as base64 string, else as file path
        """
        if is_base64:
            # Decode base64 image
            try:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                self.img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if self.img is None:
                    raise ValueError("Error: Could not decode base64 image")
            except Exception as e:
                raise ValueError(f"Error decoding base64 image: {e}")
        else:
            # Read image from file path
            self.img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if self.img is None:
                raise ValueError(f"Error: Could not read image at {image_input}")

        if self.img.shape[0] == 0 or self.img.shape[1] == 0:
            raise ValueError("Error: Loaded image has zero dimensions.")

        self.original = self.img.copy()  # Keep original for debug plotting if needed
        self.binary = None  # Binarized image (ink = 255, background = 0)
        self.words = []  # List of tuples (word_img, (x, y, w, h))
        self.results = {}  # Dictionary to hold the computed metrics
        self.all_stroke_widths = []  # List to store all measured stroke widths

    def preprocess_image(self, blur_ksize=5, threshold_block_size=15, threshold_C=5, morph_ksize=3):
        """
        Apply Gaussian blur, adaptive thresholding, and morphological operations.
        Ensures ink is white (255) and background is black (0).
        """
        if self.img is None or self.img.size == 0:
            raise ValueError("Cannot preprocess an empty image.")

        blur = cv2.GaussianBlur(self.img, (blur_ksize, blur_ksize), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, threshold_block_size, threshold_C
        )
        kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        self.binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    def segment_text_lines(self, min_line_height=10):
        """
        Segment the image into text lines using the horizontal projection profile.
        Returns a list of tuples with (line_start, line_end).
        """
        if self.binary is None or self.binary.size == 0:
            raise ValueError("Binary image not available for line segmentation.")

        horizontal_projection = np.sum(self.binary, axis=1)

        # Handle case where the image is entirely black after preprocessing
        if np.max(horizontal_projection) == 0:
            return []

        line_threshold = np.max(horizontal_projection) * 0.05
        lines = []
        in_line = False
        line_start = 0

        for i, proj in enumerate(horizontal_projection):
            is_above_threshold = proj > line_threshold
            if not in_line and is_above_threshold:
                in_line = True
                line_start = i
            elif in_line and not is_above_threshold:
                in_line = False
                if i - line_start >= min_line_height:
                    lines.append((line_start, i))
        # Add the last line if it extends to the image bottom
        if in_line and len(horizontal_projection) - line_start >= min_line_height:
            lines.append((line_start, len(horizontal_projection) - 1))

        # Removed the commented-out line merging code block
        return lines

    def extract_words(self, min_word_width=15, min_word_height=15, word_spacing_factor=0.4):
        """
        For each detected text line, extract word regions using connected components
        and horizontal spacing. Populates self.words.
        """
        if self.binary is None:
            raise ValueError("Binary image not available for word extraction.")

        self.words = []  # Clear previous words
        lines = self.segment_text_lines()
        v_buffer = 2  # Small vertical buffer around lines
        min_comp_area = 5  # Min pixel area for a connected component

        for line_start, line_end in lines:
            line_img_orig_y_start = max(0, line_start - v_buffer)
            line_img_orig_y_end = min(self.binary.shape[0], line_end + v_buffer)
            line_img = self.binary[line_img_orig_y_start:line_img_orig_y_end, :]

            if line_img.size == 0: continue

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                line_img, connectivity=8
            )

            valid_components = []
            for i in range(1, num_labels):  # Skip background label 0
                if stats[i, cv2.CC_STAT_AREA] >= min_comp_area:
                    valid_components.append({
                        'id': i,
                        'x': stats[i, cv2.CC_STAT_LEFT],
                        'y': stats[i, cv2.CC_STAT_TOP],
                        'w': stats[i, cv2.CC_STAT_WIDTH],
                        'h': stats[i, cv2.CC_STAT_HEIGHT]
                    })

            if not valid_components: continue

            sorted_components = sorted(valid_components, key=lambda c: c['x'])

            word_groups = []
            if sorted_components:  # Check if list is not empty
                current_group = [sorted_components[0]]
                for i in range(1, len(sorted_components)):
                    curr_comp = sorted_components[i]
                    prev_comp = sorted_components[i - 1]
                    gap = curr_comp['x'] - (prev_comp['x'] + prev_comp['w'])
                    space_threshold = max(10, prev_comp['w'] * word_spacing_factor)

                    if gap < space_threshold:
                        current_group.append(curr_comp)
                    else:
                        if current_group: word_groups.append(current_group)
                        current_group = [curr_comp]
                if current_group: word_groups.append(current_group)  # Add the last group

            for group in word_groups:
                if not group: continue

                min_x = min(c['x'] for c in group)
                min_y = min(c['y'] for c in group)
                max_x = max(c['x'] + c['w'] for c in group)
                max_y = max(c['y'] + c['h'] for c in group)
                word_w = max_x - min_x
                word_h = max_y - min_y

                if word_w < min_word_width or word_h < min_word_height: continue

                # Extract clean word image using component labels
                word_mask = np.zeros_like(line_img)
                for comp in group:
                    word_mask[labels == comp['id']] = 255
                clean_word_img = word_mask[min_y:max_y, min_x:max_x]

                abs_x = min_x
                abs_y = min_y + line_img_orig_y_start

                if clean_word_img.shape[0] > 0 and clean_word_img.shape[1] > 0 and np.any(clean_word_img):
                    self.words.append((clean_word_img.copy(), (abs_x, abs_y, word_w, word_h)))

    def compute_stroke_consistency(self):
        """
        Compute stroke width consistency using skeletonization and distance transform.
        Returns a dictionary of results.
        """
        self.all_stroke_widths = []
        word_mean_widths = []
        word_std_devs = []
        processed_word_count = 0

        if not self.words:
            print("Warning: No words found to analyze.")
            self.results = {
                "global_mean_stroke_width": 0.0, "global_std_dev_stroke_width": 0.0,
                "avg_word_mean_stroke_width": 0.0, "avg_word_std_dev_stroke_width": 0.0,
                "word_count": 0, "processed_word_count": 0
            }
            return self.results

        for word_idx, (word_img, _) in enumerate(self.words):
            if word_img is None or word_img.size == 0 or not np.any(word_img):
                # print(f"Warning: Skipping empty/blank word image at index {word_idx}.")
                continue

            # Ensure binary format (0 or 255)
            if not ((word_img == 0) | (word_img == 255)).all():
                _, word_img_bin = cv2.threshold(word_img, 127, 255, cv2.THRESH_BINARY)
            else:
                word_img_bin = word_img

            # 1. Skeletonization
            word_bool = word_img_bin > 0
            if not np.any(word_bool):  # Skip if no ink pixels
                continue
            try:
                skeleton_img = skeletonize(word_bool)
            except Exception as e:
                print(f"Error during skeletonization for word {word_idx}: {e}")
                continue

            if not np.any(skeleton_img):
                # print(f"Info: Empty skeleton for word {word_idx}. Skipping.")
                continue

            # 2. Distance Transform
            dist_transform = cv2.distanceTransform(word_img_bin, cv2.DIST_L2, 3)

            # 3. Width Measurement at Skeleton
            stroke_radii = dist_transform[skeleton_img]  # Radii at skeleton points
            stroke_radii = stroke_radii[stroke_radii > 1e-5]  # Filter near-zero radii

            if stroke_radii.size == 0:
                # print(f"Info: No valid stroke radii found for word {word_idx}.")
                continue

            stroke_widths = 2 * stroke_radii  # Width = 2 * Radius

            # 4. Statistics for the word
            word_mean_widths.append(np.mean(stroke_widths))
            word_std_devs.append(np.std(stroke_widths))
            self.all_stroke_widths.extend(stroke_widths.tolist())
            processed_word_count += 1

        # --- Compute Global Statistics ---
        global_mean = np.mean(self.all_stroke_widths) if self.all_stroke_widths else 0.0
        global_std = np.std(self.all_stroke_widths) if self.all_stroke_widths else 0.0
        avg_word_mean = np.mean(word_mean_widths) if word_mean_widths else 0.0
        avg_word_std = np.mean(word_std_devs) if word_std_devs else 0.0

        # --- Calculate Consistency Index ---
        consistency_index = 0.0  # Default for no strokes or zero mean
        coefficient_of_variation = 0.0
        if global_mean > 1e-6:  # Avoid division by zero or near-zero
            coefficient_of_variation = global_std / global_mean
            # Index where higher = more consistent (1 - CV, clipped)
            # If CV is small(e.g., 0.1 for very consistent strokes), the index will be high (e.g., 1.0 - 0.1 = 0.9)
            # If CV is large (e.g., 0.4 for inconsistent strokes), the index will be low (e.g., 1.0 - 0.4 = 0.6)
            consistency_index = max(0.0, 1.0 - coefficient_of_variation)
        elif not self.all_stroke_widths:
            # Default to 0, implying no measurable consistent strokes found.
            consistency_index = 0.0

        self.results = {
            "global_mean_stroke_width": float(global_mean),
            "global_std_dev_stroke_width": float(global_std),
            "coefficient_of_variation": float(coefficient_of_variation),  # Lower = More Consistent
            "stroke_consistency_index": float(consistency_index),  # Higher = More Consistent (using 1-CV)
            "avg_word_mean_stroke_width": float(avg_word_mean),
            "avg_word_std_dev_stroke_width": float(avg_word_std),
            "word_count": len(self.words),
            "processed_word_count": processed_word_count,
        }
        return self.results

    def analyze(self, debug=False):
        """
        Execute the entire analysis pipeline.

        Parameters:
            debug (bool): If True, generates and returns base64 encoded visualization graphs.

        Returns:
            dict: Contains 'metrics' dict and 'graphs' list (empty if debug=False).
                  On error, 'metrics' may contain an 'error' key.
        """
        analysis_output = {'metrics': {}, 'graphs': [], 'preprocessed_image': None}
        try:
            self.preprocess_image()
            self.extract_words()
            analysis_output['metrics'] = self.compute_stroke_consistency()

            if self.binary is not None:
                _, preprocessed_img_encoded = cv2.imencode('.png', self.binary)
                preprocessed_img_base64 = base64.b64encode(preprocessed_img_encoded).decode('utf-8')
                analysis_output['preprocessed_image'] = preprocessed_img_base64
        except ValueError as e:
            print(f"ValueError during analysis: {e}")
            analysis_output['metrics'] = {"error": str(e)}
            return analysis_output
        except Exception as e:
            print(f"An unexpected error occurred during analysis: {e}")
            import traceback
            traceback.print_exc()
            analysis_output['metrics'] = {"error": f"Unexpected error: {e}"}
            return analysis_output

        if debug:
            # --- Visualization (Import dependencies only if needed) ---
            try:
                import matplotlib.pyplot as plt
                from io import BytesIO

                fig = plt.figure(figsize=(12, 8))  # Adjusted size slightly

                # Plot 1: Original
                ax1 = fig.add_subplot(2, 2, 1)
                ax1.imshow(self.original, cmap='gray')
                ax1.set_title('Original Image')
                ax1.axis('off')

                # Plot 2: Binarized
                ax2 = fig.add_subplot(2, 2, 2)
                if self.binary is not None:
                    ax2.imshow(self.binary, cmap='gray')
                    ax2.set_title('Binarized Image (Ink=White)')
                else:
                    ax2.set_title('Binarized Image (Not Generated)')
                ax2.axis('off')

                # Plot 3: Detected Words
                ax3 = fig.add_subplot(2, 2, 3)
                vis_img = cv2.cvtColor(self.original, cv2.COLOR_GRAY2BGR)
                word_count = 0
                for _, (x, y, w, h) in self.words:
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    word_count += 1
                ax3.imshow(vis_img)
                ax3.set_title(f'Detected Words ({word_count})')
                ax3.axis('off')

                # Plot 4: Stroke Width Histogram
                ax4 = fig.add_subplot(2, 2, 4)
                if self.all_stroke_widths:
                    ax4.hist(self.all_stroke_widths, bins=30, color='skyblue', edgecolor='black')
                    mean_w = analysis_output['metrics'].get("global_mean_stroke_width", 0)
                    std_w = analysis_output['metrics'].get("global_std_dev_stroke_width", 0)
                    # --- MODIFIED TITLE ---
                    cv = analysis_output['metrics'].get("coefficient_of_variation", 0)
                    cons_idx = analysis_output['metrics'].get("stroke_consistency_index", 0)
                    ax4.set_title(f'Stroke Width (Mean={mean_w:.2f}, StdDev={std_w:.2f})\n'
                                  f'CV={cv:.2f}, Consistency Index={cons_idx:.2f}')
                    # --- END MODIFICATION ---
                    ax4.set_xlabel('Approx. Stroke Width (pixels)')
                    ax4.set_ylabel('Frequency')
                    ax4.axvline(mean_w, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_w:.2f}')
                    ax4.legend()
                else:
                    ax4.text(0.5, 0.5, 'No stroke widths measured', ha='center', va='center')
                    ax4.set_title('Stroke Width Distribution')
                    ax4.axis('off')

                plt.tight_layout(pad=2.0)

                # Convert plot to base64
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)  # Close the figure
                analysis_output['graphs'].append(plot_base64)

            except ImportError:
                print("Warning: Matplotlib not found. Cannot generate debug graphs.")
            except Exception as plot_err:
                print(f"Error generating debug plot: {plot_err}")
                # Continue without graph if plotting fails

        return analysis_output


if __name__ == "__main__":
    # Example usage with file paths:
    image_path = "../../atest/1.png"
    analyzer = StrokeConsistencyAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)

    print("\n===== Cursive Stroke Consistency Analysis Results =====")
    metrics_cursive = results['metrics']
    for key, value in metrics_cursive.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Display the graph if generated
    if results['graphs']:
        from PIL import Image
        import io

        print("\nDisplaying visualization for Cursive...")
        img_data_cursive = base64.b64decode(results['graphs'][0])
        img = Image.open(io.BytesIO(img_data_cursive))
        img.show()

    if results['preprocessed_image']:
        print("\nPreprocessed Image (Base64):")
        print(results['preprocessed_image'][:100] + "...")  # Print first 100 char for brevity
