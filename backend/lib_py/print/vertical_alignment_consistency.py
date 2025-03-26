import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import base64
from io import BytesIO

class VerticalAlignmentAnalyzer:
    """
    A class to measure vertical alignment consistency in handwriting images.
    This can help identify print-style writing which tends to have more consistent alignment.
    """

    def __init__(self, image_input, is_base64=True):
        """
        Initializes the VerticalAlignmentAnalyzer with either a base64 encoded image or image path.

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
        if len(self.img.shape) == 3:
            self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_img = self.img.copy()

    def _dbscan_cluster(self, values, epsilon):
        """
        A simple DBSCAN clustering implementation for 1D data.
        Clusters values that are within 'epsilon' distance of one another.

        Parameters:
            values (array-like): 1D array of numeric values.
            epsilon (float): Tolerance distance for clustering.

        Returns:
            np.ndarray: Array of cluster labels (starting from 1).
        """
        n = len(values)
        clusters = np.zeros(n, dtype=int)
        current_cluster = 0

        # Sort values to ease neighbor searching
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]

        for i in range(n):
            idx = sorted_indices[i]
            if clusters[idx] != 0:
                continue

            current_cluster += 1
            clusters[idx] = current_cluster

            # Find all points within epsilon of the current point
            queue = []
            for j in range(n):
                if abs(sorted_values[j] - sorted_values[i]) <= epsilon:
                    queue.append(j)

            # Process the queue
            while queue:
                current = queue.pop(0)
                idx_current = sorted_indices[current]
                if clusters[idx_current] == 0:
                    clusters[idx_current] = current_cluster
                    # Check neighbors of the current point
                    for j in range(n):
                        if abs(sorted_values[j] - sorted_values[current]) <= epsilon and clusters[sorted_indices[j]] == 0:
                            queue.append(j)

        return clusters

    def analyze(self, debug=False):
        """
        Analyzes the image to determine vertical alignment consistency in handwriting.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: A dictionary containing metrics and optional visualization graphs.
        """
        # Apply binary thresholding and invert (using THRESH_BINARY_INV so that ink is white)
        _, binary = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_BINARY_INV)
        binary_bool = (binary > 0).astype(np.uint8)

        # Find connected components using OpenCV
        # connectedComponentsWithStats returns: number of labels, label matrix, stats, centroids.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_bool, connectivity=8)

        valid_boxes = []
        valid_centroids = []

        # Skip the first label (background) and filter out noise (small components)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 50 and w > 2 and h > 5:
                valid_boxes.append([x, y, w, h])
                valid_centroids.append(centroids[i])

        if len(valid_boxes) == 0:
            metrics = {
                'baseline_deviation': 0,
                'xheight_deviation': 0,
                'overall_alignment_score': 0,
                'component_count': 0
            }
            result = {
                'metrics': metrics,
                'graphs': []
            }
            return result

        valid_boxes = np.array(valid_boxes)
        valid_centroids = np.array(valid_centroids)
        bottoms = valid_boxes[:, 1] + valid_boxes[:, 3]  # y + height = bottom edge
        tops = valid_boxes[:, 1]                         # top edge
        heights = valid_boxes[:, 3]

        # Cluster the bottom y-coordinates using a simple 1D DBSCAN
        bottom_clusters = self._dbscan_cluster(bottoms, epsilon=5)

        # Process each line separately
        line_metrics = []
        unique_clusters = np.unique(bottom_clusters)
        for cluster in unique_clusters:
            # Skip any noise clusters (if implemented as -1) and lines with too few components
            indices = np.where(bottom_clusters == cluster)[0]
            if len(indices) < 3:
                continue

            line_bottoms = bottoms[indices]
            line_tops = tops[indices]
            line_heights = heights[indices]

            # Calculate the median bottom (baseline) and an approximate x-height line (60% from the top)
            baseline = np.median(line_bottoms)
            xheight_line = np.median(line_tops + line_heights * 0.6)

            # Calculate absolute deviations
            baseline_deviations = np.abs(line_bottoms - baseline)
            xheight_deviations = np.abs((line_tops + line_heights * 0.6) - xheight_line)

            # Normalize by median letter height to get scale invariance
            median_height = np.median(line_heights)
            if median_height > 0:
                norm_baseline_deviations = baseline_deviations / median_height
                norm_xheight_deviations = xheight_deviations / median_height
            else:
                norm_baseline_deviations = baseline_deviations
                norm_xheight_deviations = xheight_deviations

            line_metrics.append([
                np.mean(norm_baseline_deviations),
                np.mean(norm_xheight_deviations),
                len(indices)
            ])

        line_metrics = np.array(line_metrics)
        if line_metrics.size > 0:
            # Weight the metrics by the number of components in each line
            weights = line_metrics[:, 2] / np.sum(line_metrics[:, 2])
            weighted_baseline_dev = np.sum(line_metrics[:, 0] * weights)
            weighted_xheight_dev = np.sum(line_metrics[:, 1] * weights)
            # Overall alignment score (convert deviations to a 0-1 score where 1 is perfect alignment)
            alignment_score = max(0, 1 - (weighted_baseline_dev + weighted_xheight_dev))
        else:
            weighted_baseline_dev = 0
            weighted_xheight_dev = 0
            alignment_score = 0

        metrics = {
            'baseline_deviation': weighted_baseline_dev,
            'xheight_deviation': weighted_xheight_dev,
            'overall_alignment_score': alignment_score,
            'component_count': len(valid_boxes)
        }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # Debug visualization if requested
        if debug:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # Original Image (convert BGR to RGB for correct display)
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            axs[0, 0].imshow(img_rgb)
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')

            # Binary Image
            axs[0, 1].imshow(binary_bool, cmap='gray')
            axs[0, 1].set_title('Binary Image')
            axs[0, 1].axis('off')

            # Image with bounding boxes and reference lines
            axs[1, 0].imshow(img_rgb)
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            for cluster in unique_clusters:
                indices = np.where(bottom_clusters == cluster)[0]
                if len(indices) < 3:
                    continue
                color = colors[int(cluster) % len(colors)]
                # Draw bounding boxes for the current line
                for idx in indices:
                    x, y, w, h = valid_boxes[idx]
                    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
                    axs[1, 0].add_patch(rect)
                # Draw the baseline
                line_bottoms = bottoms[indices]
                baseline = np.median(line_bottoms)
                axs[1, 0].plot([0, self.img.shape[1]], [baseline, baseline],
                               linestyle='--', color=color, linewidth=1.5)
                # Draw the approximate x-height line
                line_tops = tops[indices]
                line_heights = heights[indices]
                xheight_line = np.median(line_tops + line_heights * 0.6)
                axs[1, 0].plot([0, self.img.shape[1]], [xheight_line, xheight_line],
                               linestyle=':', color=color, linewidth=1.5)
            axs[1, 0].set_title('Character Bounding Boxes & Reference Lines')
            axs[1, 0].axis('off')

            # Bar plot for deviation metrics
            metrics_values = [weighted_baseline_dev, weighted_xheight_dev, alignment_score]
            labels = ['Baseline Dev', 'X-height Dev', 'Alignment Score']
            axs[1, 1].bar(labels, metrics_values)
            axs[1, 1].set_ylim(0, 1)
            axs[1, 1].set_title('Alignment Metrics')

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
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png'
    analyzer = VerticalAlignmentAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])
