import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class EnclosedLoopAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initialize the analyzer with either a base64 encoded image or image path.

        Parameters:
            image_input (str): Either base64 encoded image string or image file path
            is_base64 (bool): If True, image_input is treated as base64 string, else as file path
        """
        if is_base64:
            # Decode base64 image
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            self.img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if self.img is None:
                raise ValueError("Error: Could not decode base64 image")
        else:
            # Read image from file path
            self.img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if self.img is None:
                raise ValueError(f"Error: Could not read image at {image_input}")

        self.original = self.img.copy()  # Original grayscale image
        self.binary = None  # Binarized image
        self.words = []  # List of tuples (word_img, (x, y, w, h))
        self.results = {}  # Dictionary to hold the computed metrics

    def preprocess_image(self):
        """
        Apply Gaussian blur, adaptive thresholding, and morphological operations.
        """
        # Apply Gaussian blur to reduce noise
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)
        # Binarize the image with adaptive thresholding
        self.binary = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
        # Clean up the binary image with a closing morphological operation
        kernel = np.ones((2, 2), np.uint8)
        self.binary = cv2.morphologyEx(self.binary, cv2.MORPH_CLOSE, kernel)

    def segment_text_lines(self):
        """
        Segment the image into text lines using the horizontal projection profile.
        Returns a list of tuples with (line_start, line_end).
        """
        horizontal_projection = np.sum(self.binary, axis=1)
        line_threshold = np.max(horizontal_projection) * 0.1  # Adaptive threshold
        lines = []
        in_line = False
        line_start = 0

        for i, proj in enumerate(horizontal_projection):
            if not in_line and proj > line_threshold:
                in_line = True
                line_start = i
            elif in_line and proj <= line_threshold:
                in_line = False
                if i - line_start > 10:  # Minimum line height
                    lines.append((line_start, i))
        if in_line:
            lines.append((line_start, len(horizontal_projection) - 1))
        return lines

    def extract_words(self):
        """
        For each detected text line, extract word-like regions using connected components.
        Populates self.words with tuples (word_img, (x, y, w, h)).
        """
        lines = self.segment_text_lines()
        for line_start, line_end in lines:
            line_img = self.binary[line_start:line_end, :]
            # Use connected components to find potential words
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                line_img, connectivity=8
            )
            # Sort components by x-coordinate (skip the background, index 0)
            sorted_components = sorted(
                [(i, stats[i][0]) for i in range(1, num_labels)],
                key=lambda x: x[1]
            )

            word_groups = []
            if sorted_components:
                current_group = [sorted_components[0][0]]
            else:
                current_group = []

            for i in range(1, len(sorted_components)):
                curr_comp = sorted_components[i]
                prev_comp = sorted_components[i - 1]
                # Group components if the horizontal gap is small
                if curr_comp[1] - (prev_comp[1] + stats[prev_comp[0]][2]) < stats[prev_comp[0]][2] * 0.8:
                    current_group.append(curr_comp[0])
                else:
                    if current_group:
                        word_groups.append(current_group)
                    current_group = [curr_comp[0]]
            if current_group:
                word_groups.append(current_group)

            # Extract word regions from groups
            for group in word_groups:
                word_mask = np.zeros_like(line_img)
                for comp_id in group:
                    word_mask[labels == comp_id] = 255
                word_coords = np.column_stack(np.where(word_mask > 0))
                if len(word_coords) == 0:
                    continue
                y_min, x_min = np.min(word_coords, axis=0)
                y_max, x_max = np.max(word_coords, axis=0)
                # Adjust y coordinates relative to the original image
                y_min += line_start
                y_max += line_start
                word_img = self.binary[y_min:y_max + 1, x_min:x_max + 1]
                # Filter out very small regions
                if word_img.shape[0] > 10 and word_img.shape[1] > 10:
                    self.words.append((word_img, (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)))

    def compute_loopiness(self):
        """
        Compute loop metrics for each detected word and overall.
        Returns a dictionary of results.
        """
        total_outer_count = 0
        total_inner_count = 0
        word_loopiness = []

        for word_img, _ in self.words:
            contours, hierarchy = cv2.findContours(
                word_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            if hierarchy is None or len(hierarchy) == 0:
                continue

            outer_count = 0
            inner_count = 0
            for idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < 10:
                    continue
                # hierarchy[0][idx][3] == -1 indicates an outer contour
                if hierarchy[0][idx][3] == -1:
                    outer_count += 1
                else:
                    inner_count += 1

            word_loop_ratio = inner_count / outer_count if outer_count > 0 else 0
            word_loopiness.append(word_loop_ratio)
            total_outer_count += outer_count
            total_inner_count += inner_count

        global_loopiness = total_inner_count / total_outer_count if total_outer_count > 0 else 0
        avg_word_loopiness = np.mean(word_loopiness) if word_loopiness else 0
        std_loopiness = np.std(word_loopiness) if len(word_loopiness) > 1 else 0

        self.results = {
            "global_loopiness": global_loopiness,
            "avg_word_loopiness": avg_word_loopiness,
            "std_loopiness": std_loopiness,
            "inner_contour_count": total_inner_count,
            "outer_contour_count": total_outer_count,
            "word_count": len(self.words)
        }
        return word_loopiness

    def analyze(self, debug=False):
        """
        Execute the entire analysis pipeline.
        Returns a dictionary with metrics and optional visualization graphs.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Contains metrics and base64 encoded visualization graphs if debug=True
        """
        self.preprocess_image()
        self.extract_words()
        word_loopiness = self.compute_loopiness()

        result = {
            'metrics': self.results,
            'graphs': [],
            'preprocessed_image': ''
        }

        # Convert preprocessed image to base64
        _, preprocessed_img_encoded = cv2.imencode('.png', self.binary)
        preprocessed_img_base64 = base64.b64encode(preprocessed_img_encoded).decode('utf-8')
        result['preprocessed_image'] = preprocessed_img_base64

        if debug:
            # Convert original image to BGR for drawing bounding boxes
            vis_img = cv2.cvtColor(self.original, cv2.COLOR_GRAY2BGR)
            for _, (x, y, w, h) in self.words:
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Visualize a sample word to show loop detection
            sample_vis = None
            if self.words:
                sample_word_img, _ = self.words[len(self.words) // 2]
                sample_contours, sample_hierarchy = cv2.findContours(
                    sample_word_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )
                sample_vis = cv2.cvtColor(sample_word_img, cv2.COLOR_GRAY2BGR)
                for idx, cnt in enumerate(sample_contours):
                    if sample_hierarchy[0][idx][3] == -1:
                        cv2.drawContours(sample_vis, [cnt], 0, (0, 255, 0), 2)
                    else:
                        cv2.drawContours(sample_vis, [cnt], 0, (0, 0, 255), 2)

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(self.original, cmap='gray')
            plt.title('Original Image')

            plt.subplot(2, 2, 2)
            plt.imshow(self.binary, cmap='gray')
            plt.title('Binarized Image')

            plt.subplot(2, 2, 3)
            plt.imshow(vis_img)
            plt.title('Detected Words')

            if sample_vis is not None:
                sample_idx = len(self.words) // 2
                plt.subplot(2, 2, 4)
                plt.imshow(sample_vis)
                plt.title(f'Sample Word: Loopiness {word_loopiness[sample_idx]:.2f}')
            else:
                plt.subplot(2, 2, 4)
                plt.bar(['Global Loopiness', 'Avg Word Loopiness'],
                        [self.results["global_loopiness"], self.results["avg_word_loopiness"]])
                plt.title('Loopiness Metrics')

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
    # Example with file path
    image_path = "../../atest/cursive2.jpg"
    analyzer = EnclosedLoopAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])