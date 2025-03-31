import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class LetterDiscretenessAnalyzer:
    """
    Analyzes the discreteness of letters in handwriting, focusing on the
    consistency of spacing between adjacent letters within the same word.
    Includes pre-checks for continuous script and handles line/word grouping.
    """

    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer by loading the image.
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
        self._reset_analysis_data()  # Initialize all data attributes

    def _reset_analysis_data(self):
        """
        Clears intermediate data from previous analysis runs.
        """
        self.gray_image = None
        self.binary_image = None
        self.letter_contours = []
        self.letter_bboxes = []  # List of (x, y, w, h) tuples
        self.lines = []  # List of lists, each inner list contains indices of letter_bboxes on that line
        self.words = []  # List of lists, each inner list contains indices of letter_bboxes in that word
        self.inter_letter_spaces = []  # List of horizontal distances between adjacent letters in words
        self.is_likely_continuous = False
        # Store grouped boxes for visualization
        self.grouped_bboxes_for_vis = {'lines': [], 'words': []}

    def _preprocess_image(self):
        """
        Applies preprocessing: Grayscale, Blur, Adaptive Thresholding.
        Stores results in self.binary_image.
        Uses fixed internal parameters.
        """
        # --- Fixed Preprocessing Parameters ---
        BLUR_KSIZE = 5  # Kernel size for Gaussian blur
        ADAPTIVE_BLOCK_SIZE = 11  # Size of the neighborhood area for adaptive thresholding
        ADAPTIVE_C = 2  # Constant subtracted from the mean/weighted mean

        # 1. Grayscale
        if len(self.img_color.shape) == 3:
            self.gray_image = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_image = self.img_color.copy()  # Already grayscale
        processed = self.gray_image.copy()

        # 2. Gaussian Blur (Optional but recommended)
        if BLUR_KSIZE > 1:
            # Ensure kernel size is odd
            ksize = BLUR_KSIZE if BLUR_KSIZE % 2 != 0 else BLUR_KSIZE + 1
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

        # 3. Adaptive Thresholding (Good for varying illumination)
        self.binary_image = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                  ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)

    def _check_for_continuity(self):
        """
        Analyzes initial contours to detect likely continuous script based on
        the relative size of the largest contour. Sets self.is_likely_continuous.
        Uses fixed internal threshold. (Similar to AspectRatioAnalyzer)
        """
        # --- Continuity Check Parameter ---
        # If the longest contour perimeter is > 30% of the total perimeter,
        # assume it's likely continuous script.
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
        Finds and filters contours to isolate potential single letters (discrete components).
        Uses fixed internal constants for filtering criteria (area, height, aspect ratio).
        Skips if continuity check flag is True.
        !! ADJUSTED PARAMETERS for better robustness !!
        """
        # --- Fixed Filtering Parameters (ADJUSTED VALUES) ---
        # These may still need tuning based on typical input image resolution and text size
        MIN_AREA = 30  # Min pixel area (filters noise, small dots)
        MAX_AREA_RATIO = 0.20  # Max area relative to image (filters large blobs)
        MIN_HEIGHT = 30  # Min pixel height (filters short noise/dots)
        # MAX_HEIGHT_RATIO = 1.0          # Increased significantly: Allow taller letters
        ASPECT_RATIO_RANGE = (
        0.05, 5.0)  # Broadened significantly: Allow very thin (like 'I','l') and very wide ('M','W')

        # --- Check continuity flag first ---
        if self.is_likely_continuous:
            # print("Skipping letter candidate finding due to likely continuous script.")
            self.letter_contours = []
            self.letter_bboxes = []
            return  # Exit the method early

        if self.binary_image is None:
            print("Warning: Binary image not available for finding contours.")
            return

        contours, _ = cv2.findContours(self.binary_image.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)  # Use copy()

        self.letter_contours = []
        self.letter_bboxes = []
        image_area = self.original_height * self.original_width
        max_abs_area = image_area * MAX_AREA_RATIO
        # max_abs_height = self.original_height * MAX_HEIGHT_RATIO
        count_initial = len(contours)
        # Debugging: Track filter counts
        # count_filtered_area_min=0; count_filtered_area_max=0; count_filtered_height_min=0
        # count_filtered_height_max=0; count_filtered_aspect=0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by Area
            if area < MIN_AREA:
                # count_filtered_area_min += 1
                continue
            if area > max_abs_area:
                # count_filtered_area_max += 1
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Filter by Height
            if h < MIN_HEIGHT:
                # count_filtered_height_min += 1
                continue
            # if h > max_abs_height:
            #     # count_filtered_height_max += 1
            #     continue
            # Ensure width/height > 0 for aspect ratio calculation
            if w == 0 or h == 0:
                continue

            aspect_ratio = w / float(h)
            # Filter by Aspect Ratio
            if not (ASPECT_RATIO_RANGE[0] <= aspect_ratio <= ASPECT_RATIO_RANGE[1]):
                # count_filtered_aspect += 1
                continue

            # Passed all filters - considered a potential letter candidate
            self.letter_contours.append(cnt)
            self.letter_bboxes.append(
                {'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + w / 2, 'cy': y + h / 2})  # Store as dict

        # Debugging Print Statement
        # print(f"Initial contours: {count_initial}")
        # print(f"Filtered - Area Min: {count_filtered_area_min}, Area Max: {count_filtered_area_max}")
        # print(f"Filtered - Height Min: {count_filtered_height_min}, Height Max: {count_filtered_height_max}")
        # print(f"Filtered - Aspect Ratio: {count_filtered_aspect}")
        # print(f"Found {len(self.letter_bboxes)} potential letter candidates.")

    def _group_into_lines_and_words(self):
        """
        Groups detected letter bounding boxes into lines and words based on
        proximity using fixed internal parameters.
        Handles cases where no letters are found or script is continuous.
        """
        # --- Fixed Grouping Parameters ---
        # Factor of average height seems reasonable.
        LINE_PROXIMITY_FACTOR = 0.7  # Allows overlap up to 70% of avg height difference
        # How much horizontal gap defines a new word?
        # Factor of average width seems reasonable.
        WORD_GAP_FACTOR = 0.8  # Gaps larger than 80% of avg width start new word

        if not self.letter_bboxes or self.is_likely_continuous:
            # print("Skipping grouping: No letters found or likely continuous script.")
            self.lines = []
            self.words = []
            return

        # --- Calculate average dimensions for thresholds ---
        if len(self.letter_bboxes) > 0:
            avg_h = np.mean([b['h'] for b in self.letter_bboxes])
            avg_w = np.mean([b['w'] for b in self.letter_bboxes])
        else:
            # Avoid division by zero if no boxes (though caught earlier)
            avg_h = 10
            avg_w = 10

        line_proximity_threshold = avg_h * LINE_PROXIMITY_FACTOR
        word_gap_threshold = avg_w * WORD_GAP_FACTOR

        # --- 1. Sort boxes primarily by Y, secondarily by X ---
        # Store original index along with box data
        indexed_bboxes = [{'idx': i, **box} for i, box in enumerate(self.letter_bboxes)]
        # Sort by center Y, then center X
        sorted_bboxes = sorted(indexed_bboxes, key=lambda b: (b['cy'], b['cx']))

        # --- 2. Group into Lines ---
        self.lines = []
        if not sorted_bboxes:
            return  # No boxes to group

        current_line = [sorted_bboxes[0]]  # Start with the first box
        last_box_in_line = sorted_bboxes[0]

        for i in range(1, len(sorted_bboxes)):
            current_box = sorted_bboxes[i]
            # Check vertical proximity (using center y coordinates)
            # If vertical distance is small enough, it's on the same line
            if abs(current_box['cy'] - last_box_in_line['cy']) < line_proximity_threshold:
                current_line.append(current_box)
                # Update last box based on x-position for within-line sorting later
                last_box_in_line = max(current_line, key=lambda b: b['cx'])
            else:
                # Start a new line
                # Sort the completed line by X before storing
                current_line.sort(key=lambda b: b['cx'])
                self.lines.append(current_line)
                current_line = [current_box]
                last_box_in_line = current_box

        # Add the last line
        if current_line:
            current_line.sort(key=lambda b: b['cx'])
            self.lines.append(current_line)

        # Store line indices for metrics
        self.grouped_bboxes_for_vis['lines'] = [[b['idx'] for b in line] for line in self.lines]
        # print(f"Grouped into {len(self.lines)} lines.")

        # --- 3. Group into Words within each Line ---
        self.words = []
        for line in self.lines:
            if not line: continue  # Skip empty lines

            current_word = [line[0]]  # Start word with the first box on the line
            for i in range(1, len(line)):
                prev_box = line[i - 1]
                current_box = line[i]

                # Calculate horizontal gap between right edge of prev and left edge of current
                gap = current_box['x'] - (prev_box['x'] + prev_box['w'])

                if gap < word_gap_threshold:
                    # Small gap, same word
                    current_word.append(current_box)
                else:
                    # Large gap, start new word
                    self.words.append(current_word)
                    current_word = [current_box]

            # Add the last word of the line
            if current_word:
                self.words.append(current_word)

        # Store word indices for metrics
        self.grouped_bboxes_for_vis['words'] = [[b['idx'] for b in word] for word in self.words]
        # print(f"Grouped into {len(self.words)} words.")

    def _calculate_inter_letter_spaces(self):
        """
        Calculates the horizontal space between adjacent letter candidates
        within the same identified word. Filters out negative spaces (overlaps).
        """
        self.inter_letter_spaces = []
        if not self.words or self.is_likely_continuous:
            return

        for word in self.words:
            # Need at least two letters in a word to have a space between them
            if len(word) > 1:
                # Iterate through adjacent pairs in the word (already sorted by X)
                for i in range(len(word) - 1):
                    prev_box = word[i]
                    current_box = word[i + 1]

                    # Space = start of current box - end of previous box
                    space = current_box['x'] - (prev_box['x'] + prev_box['w'])

                    # Only consider positive spacing (actual gaps)
                    # Can be slightly negative due to bounding box inaccuracies or slight tilt
                    # A small tolerance might be needed, or just filter negatives.
                    if space > 0:
                        self.inter_letter_spaces.append(space)
                    # else:
                    #     print(f"Debug: Non-positive space detected ({space:.1f}) between boxes {prev_box['idx']} and {current_box['idx']}")

        # print(f"Calculated {len(self.inter_letter_spaces)} inter-letter spaces.")

    def _calculate_statistics(self):
        """
        Calculates summary statistics for the inter-letter spaces.
        """
        num_spaces = len(self.inter_letter_spaces)
        num_letters = len(self.letter_bboxes)
        num_lines = len(self.grouped_bboxes_for_vis['lines'])
        num_words = len(self.grouped_bboxes_for_vis['words'])

        if num_spaces > 1:  # Need at least 2 spaces for standard deviation
            spaces_array = np.array(self.inter_letter_spaces)
            mean_space = np.mean(spaces_array)
            median_space = np.median(spaces_array)
            std_dev_space = np.std(spaces_array)
            # Coefficient of Variation: std_dev / mean (relative std dev)
            cv_space = std_dev_space / mean_space if mean_space != 0 else 0
            min_space = np.min(spaces_array)
            max_space = np.max(spaces_array)
        elif num_spaces == 1:
            mean_space = median_space = min_space = max_space = self.inter_letter_spaces[0]
            std_dev_space = 0
            cv_space = 0
        else:
            mean_space, median_space, std_dev_space, cv_space, min_space, max_space = 0, 0, 0, 0, 0, 0

        metrics = {
            'is_likely_continuous': self.is_likely_continuous,
            'num_letter_candidates': num_letters,
            'num_lines_detected': num_lines,
            'num_words_detected': num_words,
            'num_inter_letter_spaces': num_spaces,
            'mean_inter_letter_space': mean_space,
            'median_inter_letter_space': median_space,
            'std_dev_inter_letter_space': std_dev_space,
            'cv_inter_letter_space': cv_space,  # Lower CV indicates more consistent spacing
            'min_inter_letter_space': min_space,
            'max_inter_letter_space': max_space,
        }
        return metrics

    def _generate_visualization(self, metrics):
        """ Generates visualization plots. """
        # --- Fixed Visualization Parameters ---
        FIGURE_SIZE = (14, 10)
        BOX_THICKNESS = 1
        # Use different colors for boxes in different words for clarity
        WORD_COLORS = plt.cm.viridis(np.linspace(0, 1, max(1, len(self.words)))) * 255  # BGR format
        WORD_COLORS = WORD_COLORS[:, ::-1].astype(np.uint8)  # Reverse to BGR and convert type

        graphs = []
        img_rgb = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        plt.figure("Letter Discreteness Analysis", figsize=FIGURE_SIZE)

        # Plot 1: Original
        plt.subplot(2, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis('off')

        # Plot 2: Preprocessed
        plt.subplot(2, 2, 2)
        if self.binary_image is not None:
            plt.imshow(self.binary_image, cmap='gray')
            plt.title("Preprocessed (Binary)")
        else:
            plt.text(0.5, 0.5, "Preprocessing Failed", ha='center', va='center')
            plt.title("Preprocessed")
        plt.axis('off')

        # Plot 3: Detected Letters & Words
        plt.subplot(2, 2, 3)
        img_with_boxes = img_rgb.copy()
        num_letters_found = metrics.get('num_letter_candidates', 0)
        if num_letters_found > 0 and not self.is_likely_continuous:
            if not self.words:  # Draw individual boxes if words couldn't be formed
                for i in range(len(self.letter_bboxes)):
                    box = self.letter_bboxes[i]
                    cv2.rectangle(img_with_boxes, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']),
                                  (0, 255, 0), BOX_THICKNESS)  # Green for all
            else:  # Color boxes by word
                for i, word in enumerate(self.words):
                    color = tuple(map(int, WORD_COLORS[i % len(WORD_COLORS)]))  # Cycle through colors
                    for box_info in word:
                        box = self.letter_bboxes[box_info['idx']]  # Get full box info using index
                        cv2.rectangle(img_with_boxes, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']),
                                      color, BOX_THICKNESS)

            title = f"Detected Letters ({num_letters_found}), Words ({metrics.get('num_words_detected', 0)})"
        elif self.is_likely_continuous:
            title = "Detected Likely Continuous Script - No Letters/Words Marked"
        else:
            title = "No Letter Candidates Found"

        plt.imshow(img_with_boxes)
        plt.title(title)
        plt.axis('off')

        # Plot 4: Inter-Letter Space Histogram
        plt.subplot(2, 2, 4)
        num_spaces = metrics.get('num_inter_letter_spaces', 0)
        if num_spaces > 0:
            # Choose number of bins, ensure it's at least 1
            num_bins = min(25, max(1, num_spaces // 2))
            plt.hist(self.inter_letter_spaces, bins=num_bins, color='skyblue', edgecolor='black')

            mean_val = metrics['mean_inter_letter_space']
            median_val = metrics['median_inter_letter_space']
            std_dev = metrics['std_dev_inter_letter_space']
            cv_val = metrics['cv_inter_letter_space']

            plt.axvline(mean_val, color='r', ls='--', lw=1.5, label=f"Mean: {mean_val:.1f}px")
            plt.axvline(median_val, color='g', ls=':', lw=1.5, label=f"Median: {median_val:.1f}px")
            plt.legend(fontsize='small')
            plt.title(f"Inter-Letter Spacing (StdDev: {std_dev:.2f}, CV: {cv_val:.3f})")
            plt.xlabel("Space (pixels)")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        elif self.is_likely_continuous:
            plt.text(0.5, 0.5, "Analysis skipped\n(likely continuous script)", ha='center', va='center',
                     transform=plt.gca().transAxes)
            plt.title("Inter-Letter Spacing")
        else:
            plt.text(0.5, 0.5, "No inter-letter spaces\ncalculated", ha='center', va='center',
                     transform=plt.gca().transAxes)
            plt.title("Inter-Letter Spacing")
        plt.tight_layout(pad=1.5)

        # Save plot to buffer and encode
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()  # Close the figure explicitly to free memory
        graphs.append(plot_base64)
        return graphs

    def analyze(self, debug=False):
        """
        Orchestrates the analysis pipeline: preprocess, check continuity,
        find letters, group lines/words, calculate spaces, compute stats.

        Returns:
            dict: Contains 'metrics' dictionary and optionally 'graphs' list.
        """
        # 0. Reset
        self._reset_analysis_data()

        # 1. Preprocess
        self._preprocess_image()

        # 2. Check for Continuity
        self._check_for_continuity()  # Sets self.is_likely_continuous

        # 3. Find Letter Candidates (skips if continuous)
        self._find_letter_candidates()

        # 4. Group into Lines and Words (skips if continuous or no letters)
        self._group_into_lines_and_words()

        # 5. Calculate Inter-Letter Spaces (skips if continuous or no words/letters)
        self._calculate_inter_letter_spaces()

        # 6. Calculate Statistics
        metrics = self._calculate_statistics()

        # 7. Prepare Result
        result = {
            'metrics': metrics,
            'graphs': [],
            'preprocessed_image': None,
        }

        # Encode preprocessed image if successful
        if self.binary_image is not None:
            _, preprocessed_buffer = cv2.imencode('.png', self.binary_image)
            result['preprocessed_image'] = base64.b64encode(preprocessed_buffer).decode('utf-8')

        # 8. Generate Visualization (Optional)
        # Generate graphs even if continuous, to show original/preprocessed and continuity status
        if debug:
            result['graphs'] = self._generate_visualization(metrics=metrics)

        return result


# === Example Usage ===
if __name__ == "__main__":
    # --- Configuration ---
    image_path = r"C:\Users\Samson\Desktop\Coding\IPPR\NoteMercy_Extension\backend\atest\print2.png"  # Example path
    analyzer = LetterDiscretenessAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    # print(results)

    # Print metrics in a readable format
    print("\n===== Letter Discreteness Analysis Results =====")
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
