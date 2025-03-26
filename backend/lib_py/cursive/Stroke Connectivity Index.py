import cv2
import numpy as np
import matplotlib.pyplot as plt

class StrokeConnectivityAnalyzer:
    def __init__(self, image_path, debug=False):
        """
        Initializes the analyzer with the path to the image and an optional debug flag.
        """
        self.image_path = image_path
        self.debug = debug

    def compute(self):
        """
        Computes stroke connectivity metrics from the image specified at initialization.
        Returns a dictionary with various metrics.
        """
        # 1. Read the image in grayscale
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read the image at: {self.image_path}")
            return None

        # Save original image for visualization if debugging
        original = img.copy() if self.debug else None

        # 2. Preprocessing: Gaussian blur, adaptive thresholding, and morphological closing
        img = cv2.GaussianBlur(img, (5, 5), 0)
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 3. Improved word segmentation using horizontal projection
        horizontal_projection = np.sum(binary, axis=1)
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
        # Handle case where the last line extends to the bottom of the image
        if in_line:
            lines.append((line_start, len(horizontal_projection) - 1))

        # For each line, segment into words using vertical projection and connected components
        words = []
        for line_start, line_end in lines:
            line_img = binary[line_start:line_end, :]

            # Use connected components with stats to find potential words
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                line_img, connectivity=8
            )

            # Group nearby components as potential words
            sorted_components = sorted(
                [(i, stats[i][0]) for i in range(1, num_labels)],
                key=lambda x: x[1]
            )
            word_groups = []
            current_group = [sorted_components[0][0]] if sorted_components else []

            for i in range(1, len(sorted_components)):
                curr_comp = sorted_components[i]
                prev_comp = sorted_components[i - 1]

                # If the horizontal distance between components is small, group them
                if curr_comp[1] - (prev_comp[1] + stats[prev_comp[0]][2]) < stats[prev_comp[0]][2] * 0.8:
                    current_group.append(curr_comp[0])
                else:
                    if current_group:
                        word_groups.append(current_group)
                    current_group = [curr_comp[0]]
            if current_group:
                word_groups.append(current_group)

            # Extract words from grouped components
            for group in word_groups:
                word_mask = np.zeros_like(line_img)
                for comp_id in group:
                    word_mask[labels == comp_id] = 255

                word_coords = np.column_stack(np.where(word_mask > 0))
                if len(word_coords) == 0:
                    continue

                y_min, x_min = np.min(word_coords, axis=0)
                y_max, x_max = np.max(word_coords, axis=0)
                # Add global line offset to y coordinates
                y_min += line_start
                y_max += line_start

                # Extract the word image from the binary image
                word_img = binary[y_min:y_max + 1, x_min:x_max + 1]

                # Filter out very small regions that might be noise
                if word_img.shape[0] > 10 and word_img.shape[1] > 10:
                    words.append((word_img, (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)))

        if not words:
            print("No word-like regions found in the image.")
            return None

        # 4. Compute stroke connectivity metrics
        component_counts = []
        character_counts = []  # Estimate of character count based on width

        for word_img, bbox in words:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                word_img, connectivity=8
            )
            valid_components = 0
            for i in range(1, num_labels):  # Skip background
                area = stats[i][4]  # Area of the component
                if area > word_img.size * 0.005:  # Filter out small components
                    valid_components += 1

            word_width = bbox[2]
            avg_char_width = word_width / 6 if word_width > 50 else word_width / 3  # Rough estimate
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
        # Clamp to 0-1 range in case of estimation errors
        connectivity_score = max(0, min(1, connectivity_score))

        results = {
            "avg_components_per_word": avg_components_per_word,
            "components_per_char": components_per_char,
            "connectivity_index": connectivity_index,
            "connectivity_ratio": connectivity_ratio,
            "connectivity_score": connectivity_score,
            "word_count": len(words),
            "estimated_char_count": sum(character_counts)
        }

        # 5. Visualization for debugging
        if self.debug:
            vis_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            for _, (x, y, w, h) in words:
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(original, cmap='gray')
            plt.title('Original Image')

            plt.subplot(2, 2, 2)
            plt.imshow(binary, cmap='gray')
            plt.title('Binarized Image')

            plt.subplot(2, 2, 3)
            plt.imshow(vis_img)
            plt.title('Detected Words')

            plt.subplot(2, 2, 4)
            plt.bar(['Connectivity Score', 'Connectivity Index', 'Components/Char'],
                    [connectivity_score, connectivity_index, components_per_char])
            plt.title('Stroke Connectivity Metrics')

            plt.tight_layout()
            plt.savefig(f"{self.image_path}_analysis.png")
            plt.close()

        print("Stroke Connectivity Analysis Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        return results


if __name__ == "__main__":
    # Replace with your actual image file path
    image_path = "atest/3.png"
    analyzer = StrokeConnectivityAnalyzer(image_path, debug=True)
    results = analyzer.compute()
