import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class StrokeConnectivityAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the StrokeConnectivityAnalyzerRevised.

        Args:
            image_input (str): Base64 encoded image string or image file path.
            is_base64 (bool): True if image_input is base64, False if file path.
        """
        if is_base64:
            try:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if self.img is None:
                    raise ValueError("Error: Could not decode base64 image")
            except Exception as e:
                raise ValueError(f"Error processing base64 image: {e}")
        else:
            self.img = cv2.imread(image_input)
            if self.img is None:
                raise ValueError(f"Error: Could not read image at {image_input}")

        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        elif len(self.img.shape) == 2:
            self.gray_img = self.img.copy()
        else:
             raise ValueError("Unsupported image format (must be color or grayscale)")

        # Estimate average stroke height for filtering (can be refined)
        self.avg_stroke_height = self._estimate_stroke_height()

    def _estimate_stroke_height(self):
        # Basic estimation - can be improved with more robust methods
        # Threshold and find contours to get an idea of typical heights
        _, thresh = cv2.threshold(self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heights = [cv2.boundingRect(c)[3] for c in contours if cv2.contourArea(c) > 5] # Ignore tiny contours
        if not heights:
            return 10 # Default fallback
        # Use median height, more robust to outliers than mean
        median_h = np.median(heights)
        # Clip to a reasonable range
        return max(5, min(median_h, 50)) # Assume stroke height is typically between 5 and 50 pixels


    def _preprocess_image(self, adaptive_block_size=15, adaptive_c=5):
        """
        Preprocesses the grayscale image for analysis.

        Args:
            adaptive_block_size (int): Block size for adaptive thresholding (must be odd).
            adaptive_c (int): Constant subtracted from the mean in adaptive thresholding.

        Returns:
            numpy.ndarray: The preprocessed binary image (text is white).
        """
        # 1. Apply Gaussian blur - moderate blur
        blurred = cv2.GaussianBlur(self.gray_img, (3, 3), 0)

        # 2. Apply adaptive thresholding - parameters might need tuning per dataset
        # Using a slightly larger block size might help with uneven illumination
        # The constant C controls how much lighter than the neighbourhood pixels need to be
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_c)

        # 3. Optional: Morphological Opening to remove small noise/dots
        # Kernel size should be smaller than the details you want to keep (like i-dots)
        # Using a 2x2 kernel might remove very small noise without damaging larger structures
        # kernel = np.ones((2, 2), np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # DO NOT USE CLOSING here as it artificially merges components.

        return binary

    def _segment_lines(self, binary_img):
        """
        Segments the binary image into horizontal text lines.
        Basic horizontal projection method.

        Args:
            binary_img (numpy.ndarray): Binarized image (text=white).

        Returns:
            list: A list of tuples [(line_start_y, line_end_y), ...].
        """
        if binary_img is None or binary_img.size == 0:
            return []

        # Horizontal projection profile
        h_proj = np.sum(binary_img, axis=1)

        # Basic thresholding to find text regions - may need tuning
        # Use Otsu on the projection profile itself or a percentile?
        proj_thresh = np.mean(h_proj[h_proj > 0]) * 0.1 if np.any(h_proj > 0) else 0 # Threshold based on non-zero rows mean

        lines = []
        in_line = False
        line_start = 0
        min_line_height = max(5, int(self.avg_stroke_height * 0.5)) # Minimum height based on estimated stroke

        for i, proj_val in enumerate(h_proj):
            is_above_thresh = proj_val > proj_thresh
            if not in_line and is_above_thresh:
                in_line = True
                line_start = i
            elif in_line and not is_above_thresh:
                in_line = False
                line_end = i
                # Check minimum line height before adding
                if (line_end - line_start) >= min_line_height:
                    lines.append((line_start, line_end))
            # Handle case where line goes to the bottom edge
            if in_line and i == len(h_proj) - 1:
                 line_end = i + 1
                 if (line_end - line_start) >= min_line_height:
                     lines.append((line_start, line_end))

        return lines


    def _segment_words(self, binary_img, lines, gap_threshold_factor=1.0):
        """
        Segments lines into words based on horizontal gaps between connected components.

        Args:
            binary_img (numpy.ndarray): Binarized image (text=white).
            lines (list): List of (y_start, y_end) tuples for each line.
            gap_threshold_factor (float): Multiplier for average component width to determine word gap.

        Returns:
            list: A list of tuples [(word_img, (x, y, w, h)), ...].
        """
        words = []
        min_component_area = max(5, int( (self.avg_stroke_height / 4)**2 )) # Min area relative to stroke height squared

        for line_start, line_end in lines:
            if line_end <= line_start: continue
            line_img = binary_img[line_start:line_end, :]
            if line_img.size == 0: continue

            # Find connected components within the line image
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                line_img, connectivity=8 # Use 8-connectivity
            )

            # Filter out background label (0) and tiny components
            valid_components = []
            component_widths = []
            for i in range(1, num_labels): # Start from 1 to ignore background
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_component_area:
                    # Store index, x-coordinate, width
                    valid_components.append({'id': i, 'x': stats[i, cv2.CC_STAT_LEFT], 'width': stats[i, cv2.CC_STAT_WIDTH]})
                    component_widths.append(stats[i, cv2.CC_STAT_WIDTH])

            if not valid_components: continue

            # Sort components by x-coordinate
            sorted_components = sorted(valid_components, key=lambda c: c['x'])

            # Estimate word gap threshold based on average component width within the line
            avg_comp_width = np.mean(component_widths) if component_widths else self.avg_stroke_height # Fallback
            word_gap_threshold = avg_comp_width * gap_threshold_factor

            # Group components into words
            word_groups = []
            current_group = [sorted_components[0]['id']]

            for i in range(1, len(sorted_components)):
                curr_comp = sorted_components[i]
                prev_comp = sorted_components[i - 1]

                # Calculate gap between the end of previous component and start of current
                gap = curr_comp['x'] - (prev_comp['x'] + prev_comp['width'])

                if gap < word_gap_threshold:
                    # Components are close, belong to the same word
                    current_group.append(curr_comp['id'])
                else:
                    # Gap is large, start a new word
                    if current_group:
                        word_groups.append(current_group)
                    current_group = [curr_comp['id']]

            # Add the last group
            if current_group:
                word_groups.append(current_group)

            # Extract word images and bounding boxes
            min_word_width = int(self.avg_stroke_height * 1.5) # Word should be wider than ~1.5 chars
            min_word_height = int(self.avg_stroke_height * 0.8) # Word should have min height

            for group in word_groups:
                # Create a mask for the current word group
                word_mask = np.isin(labels, group).astype(np.uint8) * 255

                # Find bounding box of the word using the mask
                contours, _ = cv2.findContours(word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue

                # Combine bounding boxes of all contours in the group
                x_coords = []
                y_coords = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    x_coords.extend([x, x + w])
                    y_coords.extend([y, y + h])

                if not x_coords or not y_coords: continue

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                word_w = x_max - x_min
                word_h = y_max - y_min

                # Filter based on word dimensions relative to stroke height
                if word_w >= min_word_width and word_h >= min_word_height:
                    # Extract the word image from the original binary line image
                    word_img_crop = line_img[y_min:y_max, x_min:x_max]

                    # Store the actual word image (cropped) and its global coordinates
                    global_y_min = y_min + line_start
                    bbox = (x_min, global_y_min, word_w, word_h)
                    words.append((word_img_crop, bbox))

        return words


    def _compute_metrics(self, words):
        """
        Computes stroke connectivity metrics from the segmented words.
        Focuses on robust metrics not reliant on character count estimation.

        Args:
            words (list): List of tuples [(word_img, (x, y, w, h)), ...].

        Returns:
            dict: Dictionary of computed metrics.
        """
        if not words:
            return {
                "average_components_per_word": 0,
                "average_component_density": 0, # Components per pixel area
                "word_count": 0,
                "status": "No words detected"
            }

        component_counts_per_word = []
        component_densities = []
        total_components = 0
        min_component_area = max(5, int((self.avg_stroke_height / 4)**2)) # Consistent filtering

        for word_img, bbox in words:
            if word_img is None or word_img.size == 0:
                continue

            # Find connected components within the isolated word image
            # Ensure image is binary
            _, word_binary = cv2.threshold(word_img, 127, 255, cv2.THRESH_BINARY)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                word_binary, connectivity=8
            )

            # Count valid components (ignore background and tiny specks)
            valid_components_count = 0
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
                    valid_components_count += 1

             # Avoid division by zero if no valid components found (should be rare for valid words)
            if valid_components_count == 0:
                 valid_components_count = 1 # Assume at least one component if word was detected

            component_counts_per_word.append(valid_components_count)
            total_components += valid_components_count

            # Calculate component density
            word_area = bbox[2] * bbox[3] # w * h
            if word_area > 0:
                density = valid_components_count / word_area
                component_densities.append(density)
            else:
                 component_densities.append(0)


        avg_components = np.mean(component_counts_per_word) if component_counts_per_word else 0
        avg_density = np.mean(component_densities) if component_densities else 0

        return {
            "average_components_per_word": avg_components,
            "average_component_density": avg_density * 1000, # Scale for readability (e.g., comps per 1000 pixels)
            "word_count": len(words),
            "total_components_found": total_components,
            "status": "Success"
        }

    def analyze(self, debug=False, adaptive_block_size=15, adaptive_c=5, gap_threshold_factor=1.0):
        """
        Analyzes the image to determine stroke connectivity characteristics.

        Args:
            debug (bool): If True, generates visualization plots.
            adaptive_block_size (int): Parameter for preprocessing.
            adaptive_c (int): Parameter for preprocessing.
            gap_threshold_factor (float): Parameter for word segmentation.

        Returns:
            dict: Contains 'metrics' and 'graphs' (if debug=True).
                  Metrics focus on `average_components_per_word`.
        """
        try:
            # Preprocess the image
            binary = self._preprocess_image(adaptive_block_size, adaptive_c)

            # Segment into lines and words
            lines = self._segment_lines(binary)
            if not lines:
                 print("Warning: No text lines detected.")
                 metrics = self._compute_metrics([]) # Return default empty metrics
                 return {'metrics': metrics, 'graphs': []}

            words = self._segment_words(binary, lines, gap_threshold_factor)
            if not words:
                 print("Warning: No words detected.")
                 metrics = self._compute_metrics([]) # Return default empty metrics
                 return {'metrics': metrics, 'graphs': []}

            # Compute metrics
            metrics = self._compute_metrics(words)

            result = {'metrics': metrics, 'graphs': []}

            # --- Debug Visualization ---
            if debug:
                # Ensure img is RGB for plotting
                if len(self.img.shape) == 2:
                    img_rgb = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

                plt.figure("Stroke Connectivity Analysis (Revised)", figsize=(12, 10))

                # 1: Original Image
                plt.subplot(2, 2, 1)
                plt.imshow(img_rgb)
                plt.title("Original Image")
                plt.axis('off')

                # 2: Binarized Image
                plt.subplot(2, 2, 2)
                plt.imshow(binary, cmap='gray')
                plt.title(f"Binarized (Block:{adaptive_block_size}, C:{adaptive_c})")
                plt.axis('off')

                # 3: Detected Words
                plt.subplot(2, 2, 3)
                vis_img_words = img_rgb.copy()
                # Draw lines first (optional)
                # for y_start, y_end in lines:
                #    cv2.rectangle(vis_img_words, (0, y_start), (vis_img_words.shape[1]-1, y_end), (255, 0, 0), 1) # Blue lines
                # Draw words
                for _, (x, y, w, h) in words:
                    cv2.rectangle(vis_img_words, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green words
                plt.imshow(vis_img_words)
                plt.title(f"Detected Words ({len(words)})")
                plt.axis('off')

                # 4: Key Metrics Visualization
                plt.subplot(2, 2, 4)
                if metrics['word_count'] > 0:
                    metric_names = ['Avg Comps/Word', 'Avg Comp Density\n(per 1k px)']
                    metric_values = [metrics['average_components_per_word'], metrics['average_component_density']]
                    bars = plt.bar(metric_names, metric_values)
                    plt.bar_label(bars, fmt='{:.2f}')
                    plt.title('Key Connectivity Metrics')
                    plt.ylabel('Value')
                    # Add suggested thresholds visually? (more complex)
                    # plt.axhline(y=SUGGESTED_CURSIVE_THRESHOLD, color='r', linestyle='--', label=f'Cursive thr: {SUGGESTED_CURSIVE_THRESHOLD}')
                    # plt.legend()
                else:
                    plt.text(0.5, 0.5, "No words detected\nto compute metrics", horizontalalignment='center', verticalalignment='center')
                    plt.title('Key Connectivity Metrics')


                plt.tight_layout()

                # Save plot to base64
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                # plt.show() # Optional: display interactively
                plt.close()

                result['graphs'].append(plot_base64)

        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            result = {
                'metrics': {
                    "average_components_per_word": 0, "average_component_density": 0,
                    "word_count": 0, "status": f"Error: {e}"
                 },
                'graphs': []
            }

        return result

# --- Threshold Suggestion Function ---
def classify_connectivity(metrics, cursive_threshold=3.5, print_threshold=6.0):
    """
    Classifies handwriting style based on connectivity metrics.

    Args:
        metrics (dict): The output metrics from the analyzer.
        cursive_threshold (float): Max avg_components_per_word to be considered cursive.
        print_threshold (float): Min avg_components_per_word to be considered print.

    Returns:
        str: Classification ("Cursive", "Print", "Mixed/Unknown").
    """
    if metrics.get('status') != 'Success' or metrics['word_count'] == 0:
        return "Error/No Text"

    avg_comps = metrics['average_components_per_word']

    if avg_comps <= cursive_threshold:
        return "Cursive"
    elif avg_comps >= print_threshold:
        return "Print"
    else:
        # Values between the thresholds are ambiguous
        return "Mixed/Unknown"


# === Example usage ===
if __name__ == "__main__":
    # Example with file path - REPLACE WITH YOUR IMAGE PATH
    # image_path_cursive = '/path/to/your/cursive_example.png'
    # image_path_print = '/path/to/your/print_example.png'
    image_path = '../../atest/italic.jpg' # Use the path from user example

    print(f"Analyzing image: {image_path}")
    try:
        # You might need to tune these parameters based on your specific images
        analyzer = StrokeConnectivityAnalyzer(image_path, is_base64=False)
        results = analyzer.analyze(
            debug=True,
            adaptive_block_size=19, # Larger block size can sometimes help with variable lighting
            adaptive_c=7,           # Adjust C based on stroke thickness/contrast
            gap_threshold_factor=0.8 # Adjust based on typical spacing in your samples
        )

        print("\n--- Analysis Results (Revised) ---")
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"Status: {metrics.get('status', 'N/A')}")
            print(f"Word Count: {metrics.get('word_count', 0)}")
            if metrics.get('word_count', 0) > 0:
                 print(f"Average Components per Word: {metrics.get('average_components_per_word', 0):.2f}")
                 print(f"Average Component Density (per 1k px): {metrics.get('average_component_density', 0):.2f}")
                 print(f"Total Components Found: {metrics.get('total_components_found', 0)}")

                 # --- Classification Example ---
                 # **These thresholds are STARTING POINTS and likely need tuning based on testing with your data**
                 suggested_cursive_threshold = 3.5
                 suggested_print_threshold = 6.0
                 classification = classify_connectivity(metrics, suggested_cursive_threshold, suggested_print_threshold)
                 print(f"\n--- Suggested Classification ---")
                 print(f"Based on avg_comps/word (Cursive <= {suggested_cursive_threshold}, Print >= {suggested_print_threshold}): {classification}")

        else:
            print("Analysis did not return metrics.")

        # Access graph data if needed:
        # if results.get('graphs'):
        #     print("\nDebug graph generated.") # results['graphs'][0] contains the base64 string

    except ValueError as e:
        print(f"Initialization Error: {e}")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()