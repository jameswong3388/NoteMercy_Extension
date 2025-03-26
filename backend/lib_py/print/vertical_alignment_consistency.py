import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class VerticalAlignmentAnalyzer:
    """
    A class to measure vertical alignment consistency in handwriting images.
    This can help identify print-style writing which tends to have more consistent alignment.
    """

    def compute_vertical_alignment_consistency(self, image_path, debug=False):
        """
        Compute vertical alignment consistency for handwriting in an image.

        Parameters:
            image_path (str): Path to the image file.
            debug (bool): If True, display debug visualizations.

        Returns:
            dict: A dictionary with the following keys:
                - baseline_deviation: Normalized deviation of the baselines.
                - xheight_deviation: Normalized deviation of the approximate x-height.
                - overall_alignment_score: A score (0-1) where 1 indicates perfect alignment.
                - component_count: The number of valid components analyzed.
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return {}

        # Convert to grayscale if the image is colored
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Apply binary thresholding and invert (using THRESH_BINARY_INV so that ink is white)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
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
            return {
                'baseline_deviation': 0,
                'xheight_deviation': 0,
                'overall_alignment_score': 0,
                'component_count': 0
            }

        valid_boxes = np.array(valid_boxes)
        valid_centroids = np.array(valid_centroids)
        bottoms = valid_boxes[:, 1] + valid_boxes[:, 3]  # y + height = bottom edge
        tops = valid_boxes[:, 1]                         # top edge
        heights = valid_boxes[:, 3]

        # Cluster the bottom y-coordinates using a simple 1D DBSCAN
        bottom_clusters = self.dbscan_cluster(bottoms, epsilon=5)

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

        results = {
            'baseline_deviation': weighted_baseline_dev,
            'xheight_deviation': weighted_xheight_dev,
            'overall_alignment_score': alignment_score,
            'component_count': len(valid_boxes)
        }

        # Debug visualization if requested
        if debug:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # Original Image (convert BGR to RGB for correct display)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                axs[1, 0].plot([0, img.shape[1]], [baseline, baseline],
                               linestyle='--', color=color, linewidth=1.5)
                # Draw the approximate x-height line
                line_tops = tops[indices]
                line_heights = heights[indices]
                xheight_line = np.median(line_tops + line_heights * 0.6)
                axs[1, 0].plot([0, img.shape[1]], [xheight_line, xheight_line],
                               linestyle=':', color=color, linewidth=1.5)
            axs[1, 0].set_title('Character Bounding Boxes & Reference Lines')
            axs[1, 0].axis('off')

            # Bar plot for deviation metrics
            metrics = [weighted_baseline_dev, weighted_xheight_dev, alignment_score]
            labels = ['Baseline Dev', 'X-height Dev', 'Alignment Score']
            axs[1, 1].bar(labels, metrics)
            axs[1, 1].set_ylim(0, 1)
            axs[1, 1].set_title('Alignment Metrics')

            plt.tight_layout()
            plt.show()

            print(f"Baseline deviation (normalized): {weighted_baseline_dev:.3f}")
            print(f"X-height deviation (normalized): {weighted_xheight_dev:.3f}")
            print(f"Overall alignment score: {alignment_score:.3f}")
            print(f"Number of components analyzed: {len(valid_boxes)}")

        return results

    def dbscan_cluster(self, values, epsilon):
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

# === Test Section ===
if __name__ == "__main__":
    analyzer = VerticalAlignmentAnalyzer()
    # Update the image_path with the path to your test image
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png'
    results = analyzer.compute_vertical_alignment_consistency(image_path, debug=True)
    print(results)
