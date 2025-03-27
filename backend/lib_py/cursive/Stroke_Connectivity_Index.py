import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class StrokeConnectivityAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the StrokeConnectivityAnalyzer with either a base64 encoded image or image path.

        Parameters:
            image_input (str): Either base64 encoded image string or image file path
            is_base64 (bool): If True, image_input is treated as base64 string, else as file path
        """
        if is_base64:
            # Decode base64 image
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if self.img is None:
                raise ValueError("Error: Could not decode base64 image")
        else:
            # Read image from file path
            self.img = cv2.imread(image_input)
            if self.img is None:
                raise ValueError(f"Error: Could not read image at {image_input}")

        # Convert to grayscale if needed
        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_img = self.img.copy()

    def _preprocess_image(self):
        """
        Preprocesses the grayscale image for analysis.
        Returns the preprocessed binary image.
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(self.gray_img, (5, 5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological closing
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def _segment_lines(self, binary):
        """
        Segments the binary image into lines using horizontal projection.
        Returns a list of (start, end) line coordinates.
        """
        horizontal_projection = np.sum(binary, axis=1)
        line_threshold = np.max(horizontal_projection) * 0.1

        lines = []
        in_line = False
        line_start = 0

        for i, proj in enumerate(horizontal_projection):
            if not in_line and proj > line_threshold:
                in_line = True
                line_start = i
            elif in_line and proj <= line_threshold:
                in_line = False
                if i - line_start > 10:
                    lines.append((line_start, i))
        
        if in_line:
            lines.append((line_start, len(horizontal_projection) - 1))
        
        return lines

    def _segment_words(self, binary, lines):
        """
        Segments lines into words using connected components analysis.
        Returns a list of word images and their bounding boxes.
        """
        words = []
        for line_start, line_end in lines:
            line_img = binary[line_start:line_end, :]
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                line_img, connectivity=8
            )
            
            sorted_components = sorted(
                [(i, stats[i][0]) for i in range(1, num_labels)],
                key=lambda x: x[1]
            )
            
            word_groups = []
            current_group = [sorted_components[0][0]] if sorted_components else []
            
            for i in range(1, len(sorted_components)):
                curr_comp = sorted_components[i]
                prev_comp = sorted_components[i - 1]
                
                if curr_comp[1] - (prev_comp[1] + stats[prev_comp[0]][2]) < stats[prev_comp[0]][2] * 0.8:
                    current_group.append(curr_comp[0])
                else:
                    if current_group:
                        word_groups.append(current_group)
                    current_group = [curr_comp[0]]
            
            if current_group:
                word_groups.append(current_group)
            
            for group in word_groups:
                word_mask = np.zeros_like(line_img)
                for comp_id in group:
                    word_mask[labels == comp_id] = 255
                
                word_coords = np.column_stack(np.where(word_mask > 0))
                if len(word_coords) == 0:
                    continue
                
                y_min, x_min = np.min(word_coords, axis=0)
                y_max, x_max = np.max(word_coords, axis=0)
                y_min += line_start
                y_max += line_start
                
                word_img = binary[y_min:y_max + 1, x_min:x_max + 1]
                
                if word_img.shape[0] > 10 and word_img.shape[1] > 10:
                    words.append((word_img, (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)))
        
        return words

    def _compute_metrics(self, words):
        """
        Computes stroke connectivity metrics from the segmented words.
        Returns a dictionary of metrics.
        """
        if not words:
            return {
                "avg_components_per_word": 0,
                "components_per_char": 0,
                "connectivity_index": 0,
                "connectivity_ratio": 0,
                "connectivity_score": 0,
                "word_count": 0,
                "estimated_char_count": 0
            }

        component_counts = []
        character_counts = []

        for word_img, bbox in words:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                word_img, connectivity=8
            )
            valid_components = 0
            for i in range(1, num_labels):
                area = stats[i][4]
                if area > word_img.size * 0.005:
                    valid_components += 1

            word_width = bbox[2]
            avg_char_width = word_width / 6 if word_width > 50 else word_width / 3
            estimated_chars = max(1, round(word_width / avg_char_width))

            component_counts.append(valid_components)
            character_counts.append(estimated_chars)

        avg_components_per_word = np.mean(component_counts)
        components_per_char = sum(component_counts) / sum(character_counts)
        connectivity_index = (avg_components_per_word - 1) / np.mean(character_counts)
        connectivity_ratio = sum(component_counts) / len(words)
        estimated_total_chars = sum(character_counts)
        connectivity_score = 1 - ((sum(component_counts) - len(words)) /
                                (estimated_total_chars - len(words)))
        connectivity_score = max(0, min(1, connectivity_score))

        return {
            "avg_components_per_word": avg_components_per_word,
            "components_per_char": components_per_char,
            "connectivity_index": connectivity_index,
            "connectivity_ratio": connectivity_ratio,
            "connectivity_score": connectivity_score,
            "word_count": len(words),
            "estimated_char_count": sum(character_counts)
        }

    def analyze(self, debug=False):
        """
        Analyzes the image to determine stroke connectivity characteristics.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Contains metrics and visualization graphs (if debug=True)
        """
        # Preprocess the image
        binary = self._preprocess_image()
        
        # Segment into lines and words
        lines = self._segment_lines(binary)
        words = self._segment_words(binary, lines)
        
        # Compute metrics
        metrics = self._compute_metrics(words)
        
        result = {
            'metrics': metrics,
            'graphs': []
        }
        
        # Generate visualization plots if debug mode is enabled
        if debug:
            # Prepare a RGB version of the image for display
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            plt.figure("Stroke Connectivity Analysis", figsize=(15, 10))
            
            # Subplot 1: Original Image
            plt.subplot(2, 2, 1)
            plt.imshow(img_rgb)
            plt.title("Original Image")
            plt.axis('off')
            
            # Subplot 2: Binary Image
            plt.subplot(2, 2, 2)
            plt.imshow(binary, cmap='gray')
            plt.title("Binarized Image")
            plt.axis('off')
            
            # Subplot 3: Detected Words
            plt.subplot(2, 2, 3)
            vis_img = img_rgb.copy()
            for _, (x, y, w, h) in words:
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.imshow(vis_img)
            plt.title("Detected Words")
            plt.axis('off')
            
            # Subplot 4: Metrics Visualization
            plt.subplot(2, 2, 4)
            plt.bar(['Connectivity Score', 'Connectivity Index', 'Components/Char'],
                   [metrics['connectivity_score'], metrics['connectivity_index'], 
                    metrics['components_per_char']])
            plt.title('Stroke Connectivity Metrics')
            
            plt.tight_layout()
            
            # Convert plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            result['graphs'].append(plot_base64)
        
        return result


# === Example usage ===
if __name__ == "__main__":
    # Example with file path
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/3.png'
    analyzer = StrokeConnectivityAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results)
