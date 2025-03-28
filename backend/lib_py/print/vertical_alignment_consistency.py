import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import base64
from io import BytesIO
import warnings # To suppress potential division-by-zero warnings if needed

class VerticalAlignmentAnalyzer:
    """
    A class to measure vertical alignment consistency in handwriting images.
    Analyzes baseline and x-height alignment to help distinguish print-style
    writing (typically more aligned) from cursive or irregular writing.
    """

    def __init__(self, image_input, is_base64=True):
        """
        Initializes the VerticalAlignmentAnalyzer with either a base64 encoded image
        or an image file path.

        Parameters:
            image_input (str): Either base64 encoded image string or image file path.
            is_base64 (bool): If True, image_input is treated as base64 string,
                              else as file path.
        """
        try:
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
            elif len(self.img.shape) == 2:
                self.gray_img = self.img.copy()
            else:
                 raise ValueError("Error: Invalid image format/channels")

        except Exception as e:
            raise ValueError(f"Error initializing VerticalAlignmentAnalyzer: {e}")

    def _dbscan_cluster_1d(self, values, epsilon):
        """
        A simple DBSCAN clustering implementation for 1D data.
        Clusters values that are within 'epsilon' distance of one another.

        Parameters:
            values (np.ndarray): 1D array of numeric values.
            epsilon (float): Tolerance distance for clustering.

        Returns:
            np.ndarray: Array of cluster labels (starting from 1). 0 indicates noise/unclustered.
        """
        if len(values) == 0:
            return np.array([], dtype=int)

        n = len(values)
        clusters = np.zeros(n, dtype=int)
        current_cluster = 0

        # Sort values to ease neighbor searching
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]

        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            # Find neighbors
            neighbors_idx = [j for j in range(n) if abs(sorted_values[j] - sorted_values[i]) <= epsilon]

            # If not enough neighbors, mark as noise (or potential cluster start later)
            # For 1D, let's be lenient and allow single points to start a cluster search
            # if len(neighbors_idx) < min_pts:
            #    clusters[sorted_indices[i]] = 0 # Mark as noise initially
            #    continue

            # Start a new cluster
            current_cluster += 1
            clusters[sorted_indices[i]] = current_cluster

            # Expand the cluster
            queue = [idx for idx in neighbors_idx if idx != i] # Neighbors except self

            while queue:
                current_neighbor_idx = queue.pop(0)

                if not visited[current_neighbor_idx]:
                    visited[current_neighbor_idx] = True
                    # Find neighbors of the current neighbor
                    expanded_neighbors_idx = [j for j in range(n) if abs(sorted_values[j] - sorted_values[current_neighbor_idx]) <= epsilon]
                    # Add new, unvisited neighbors to the queue
                    for expanded_idx in expanded_neighbors_idx:
                         if not visited[expanded_idx]:
                              # Add to queue only if not already there? In 1D sort, less critical
                              if expanded_idx not in queue:
                                   queue.append(expanded_idx)


                # If the neighbor hasn't been assigned to a cluster yet, assign it
                if clusters[sorted_indices[current_neighbor_idx]] == 0:
                    clusters[sorted_indices[current_neighbor_idx]] = current_cluster

        return clusters


    def analyze(self,
                threshold_method=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                use_morph_open=True,
                morph_kernel_size=(2,2),
                min_area=30,
                min_width=2,
                min_height=5,
                min_aspect_ratio=0.1,
                max_aspect_ratio=10.0,
                dbscan_epsilon=None, # If None, calculated dynamically
                line_min_components=3,
                debug=False):
        """
        Analyzes the image to determine vertical alignment consistency.

        Parameters:
            threshold_method (int): OpenCV thresholding method (e.g., cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU).
            use_morph_open (bool): Whether to apply morphological opening to remove noise.
            morph_kernel_size (tuple): Kernel size for morphological opening (if used).
            min_area (int): Minimum area (pixels) for a connected component to be considered.
            min_width (int): Minimum width (pixels) for a component.
            min_height (int): Minimum height (pixels) for a component.
            min_aspect_ratio (float): Minimum aspect ratio (width/height) for a component.
            max_aspect_ratio (float): Maximum aspect ratio (width/height) for a component.
            dbscan_epsilon (float, optional): Epsilon value for DBSCAN clustering of baselines/tops.
                                               If None, it's estimated based on median component height.
            line_min_components (int): Minimum number of components required to form a valid line.
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: A dictionary containing metrics and optional visualization graphs.
                  Metrics include:
                  - 'baseline_deviation': Average normalized deviation from the median baseline.
                  - 'xheight_deviation': Average normalized deviation from the estimated x-height line.
                  - 'height_consistency': Average normalized standard deviation of component heights per line.
                  - 'overall_alignment_score': Combined score (1 - (baseline_dev + xheight_dev)), capped at 0-1.
                  - 'component_count': Number of valid components detected after filtering.
                  - 'line_count': Number of valid text lines detected.
        """
        # --- Preprocessing ---
        # Apply binary thresholding (OTSU can find a good threshold automatically)
        # THRESH_BINARY_INV makes ink white (foreground) on black background.
        _, binary = cv2.threshold(self.gray_img, 0, 255, threshold_method)

        # Optional: Morphological Opening (Erosion followed by Dilation) to remove small noise
        if use_morph_open:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        binary_bool = (binary > 0).astype(np.uint8)

        # --- Component Detection & Filtering ---
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_bool, connectivity=8)

        valid_indices = []
        initial_heights = []
        # Skip the first label (background)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            aspect_ratio = w / h if h > 0 else float('inf')

            # Apply filtering criteria
            if (area >= min_area and
                w >= min_width and
                h >= min_height and
                min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                valid_indices.append(i)
                initial_heights.append(h)

        if not valid_indices:
            # No components passed filtering
            metrics = {
                'baseline_deviation': 0,
                'xheight_deviation': 0,
                'height_consistency': 1.0, # Max inconsistency
                'overall_alignment_score': 0,
                'component_count': 0,
                'line_count': 0
            }
            return {'metrics': metrics, 'graphs': []}

        valid_stats = stats[valid_indices]
        valid_centroids = centroids[valid_indices]
        component_count = len(valid_indices)

        # Extract relevant info: bounding boxes, bottoms, tops, heights
        # Format: [x, y, w, h]
        valid_boxes = valid_stats[:, :4]
        bottoms = valid_boxes[:, 1] + valid_boxes[:, 3]  # y + height
        tops = valid_boxes[:, 1]                         # y
        heights = valid_boxes[:, 3]                      # h

        # --- Dynamic Epsilon Calculation ---
        if dbscan_epsilon is None:
            if initial_heights:
                median_h = np.median(initial_heights)
                # Heuristic: epsilon related to a fraction of median height (e.g., 20-30%)
                # Adjust this factor based on testing
                dynamic_epsilon = max(3.0, median_h * 0.25)
            else:
                dynamic_epsilon = 5.0 # Fallback default if no heights found (shouldn't happen)
        else:
            dynamic_epsilon = dbscan_epsilon

        # --- Line Segmentation (Clustering Baselines) ---
        # Cluster the bottom y-coordinates using 1D DBSCAN
        bottom_clusters = self._dbscan_cluster_1d(bottoms, epsilon=dynamic_epsilon)

        # --- Per-Line Analysis ---
        line_metrics = []
        unique_clusters = np.unique(bottom_clusters[bottom_clusters > 0]) # Ignore noise label 0
        valid_lines_data = [] # Store data for debug plot

        for cluster_label in unique_clusters:
            indices = np.where(bottom_clusters == cluster_label)[0]

            # Filter lines with too few components
            if len(indices) < line_min_components:
                continue

            line_bottoms = bottoms[indices]
            line_tops = tops[indices]
            line_heights = heights[indices]

            # 1. Baseline Calculation
            baseline = np.median(line_bottoms)
            baseline_deviations = np.abs(line_bottoms - baseline)

            # 2. X-Height Estimation (using clustering on tops)
            top_clusters = self._dbscan_cluster_1d(line_tops, epsilon=dynamic_epsilon)
            # Find the most frequent cluster label among tops (excluding noise 0)
            unique_top_clusters, counts = np.unique(top_clusters[top_clusters > 0], return_counts=True)
            if len(counts) > 0:
                dominant_top_cluster = unique_top_clusters[np.argmax(counts)]
                xheight_line = np.median(line_tops[top_clusters == dominant_top_cluster])
            else:
                # Fallback if top clustering fails: use a simpler median or fixed ratio
                xheight_line = np.median(line_tops) # Simpler fallback

            # Calculate deviations from the estimated x-height line
            # Note: This deviation measures how well the *tops* align, assuming they form the x-height.
            xheight_deviations = np.abs(line_tops - xheight_line)


            # 3. Height Consistency Calculation
            height_std_dev = np.std(line_heights)

            # 4. Normalization (using median height *of the current line*)
            # Use np.maximum to avoid division by zero if median_height is 0
            median_height = np.median(line_heights)
            with warnings.catch_warnings(): # Suppress potential RuntimeWarning for division by zero
                 warnings.simplefilter("ignore", category=RuntimeWarning)
                 norm_baseline_dev = np.mean(baseline_deviations / np.maximum(median_height, 1e-6))
                 norm_xheight_dev = np.mean(xheight_deviations / np.maximum(median_height, 1e-6))
                 norm_height_std_dev = height_std_dev / np.maximum(median_height, 1e-6)


            line_metrics.append([
                norm_baseline_dev,
                norm_xheight_dev,
                norm_height_std_dev,
                len(indices) # Store component count for weighting
            ])

            # Store data for potential debug plot
            valid_lines_data.append({
                'indices': indices,
                'baseline': baseline,
                'xheight_line': xheight_line,
                'cluster_label': cluster_label
            })

        # --- Aggregate Metrics ---
        if line_metrics:
            line_metrics_arr = np.array(line_metrics)
            weights = line_metrics_arr[:, 3] / np.sum(line_metrics_arr[:, 3]) # Weight by component count

            weighted_baseline_dev = np.sum(line_metrics_arr[:, 0] * weights)
            weighted_xheight_dev = np.sum(line_metrics_arr[:, 1] * weights)
            weighted_height_consistency = np.sum(line_metrics_arr[:, 2] * weights) # Lower is better

            # Overall alignment score (higher is better)
            # Penalized by both baseline and x-height deviations
            alignment_score = max(0.0, 1.0 - (weighted_baseline_dev + weighted_xheight_dev))
            line_count = len(line_metrics)
        else:
            # No valid lines found after filtering
            weighted_baseline_dev = 1.0 # Max deviation
            weighted_xheight_dev = 1.0 # Max deviation
            weighted_height_consistency = 1.0 # Max inconsistency
            alignment_score = 0.0
            line_count = 0

        metrics = {
            'baseline_deviation': float(weighted_baseline_dev),
            'xheight_deviation': float(weighted_xheight_dev),
            'height_consistency': float(weighted_height_consistency),
            'overall_alignment_score': float(alignment_score),
            'component_count': int(component_count),
            'line_count': int(line_count)
        }

        result = {'metrics': metrics, 'graphs': []}

        # --- Debug Visualization ---
        if debug:
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))

            # Convert BGR to RGB for matplotlib display
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

            # Original Image
            axs[0, 0].imshow(img_rgb)
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')

            # Binary Image
            axs[0, 1].imshow(binary, cmap='gray')
            axs[0, 1].set_title(f'Binary (Method: {threshold_method}, Morph: {use_morph_open})')
            axs[0, 1].axis('off')

            # Image with Bounding Boxes and Reference Lines
            axs[1, 0].imshow(img_rgb)
            # Use a colormap for potentially many lines
            cmap = plt.get_cmap('viridis', len(valid_lines_data) + 1)

            for i, line_data in enumerate(valid_lines_data):
                color = cmap(i)
                indices = line_data['indices']
                baseline = line_data['baseline']
                xheight_line = line_data['xheight_line']

                # Draw bounding boxes for the current line
                for idx in indices:
                    x, y, w, h = valid_boxes[idx]
                    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none', alpha=0.7)
                    axs[1, 0].add_patch(rect)

                # Draw the baseline
                axs[1, 0].plot([0, self.img.shape[1]], [baseline, baseline],
                               linestyle='--', color=color, linewidth=1.5, label=f'Line {i+1} Base' if i < 5 else None)
                # Draw the estimated x-height line
                axs[1, 0].plot([0, self.img.shape[1]], [xheight_line, xheight_line],
                               linestyle=':', color=color, linewidth=1.5, label=f'Line {i+1} X-H' if i < 5 else None)

            axs[1, 0].set_title(f'Comp: {component_count}, Lines: {line_count}, Eps: {dynamic_epsilon:.2f}')
            axs[1, 0].axis('off')
            # Set y-limits to image height inverted (origin top-left)
            axs[1, 0].set_ylim(self.img.shape[0], 0)
            if len(valid_lines_data) > 0 and len(valid_lines_data) <= 5 : # Add legend only if few lines
                 axs[1, 0].legend(fontsize='small', loc='upper right')


            # Bar plot for deviation metrics
            metric_labels = ['Baseline Dev', 'X-height Dev', 'Height Incons.', 'Alignment Score']
            metric_values = [metrics['baseline_deviation'], metrics['xheight_deviation'], metrics['height_consistency'], metrics['overall_alignment_score']]
            bars = axs[1, 1].bar(metric_labels, metric_values, color=['red', 'red', 'orange', 'green'])
            axs[1, 1].set_ylim(0, 1.1) # Allow slight overshoot visually
            axs[1, 1].set_ylabel("Normalized Metric Value")
            axs[1, 1].set_title('Alignment Metrics (Lower dev/incons. is better)')
            # Add value labels on bars
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center') # Add text labels

            plt.tight_layout()

            # Convert plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig) # Close the figure to free memory

            result['graphs'].append(plot_base64)

        return result

# === Example usage ===
if __name__ == "__main__":
    # Replace with a valid image path on your system
    # Use images with clear print and clear cursive/mixed writing for testing
    image_path_print = '../../atest/italic3.png' # <--- CHANGE THIS
    image_path_cursive = '../../atest/1.png' # <--- CHANGE THIS

    print("--- Analyzing Print Sample ---")
    try:
        # Example with default settings
        analyzer_print = VerticalAlignmentAnalyzer(image_path_print, is_base64=False)
        results_print = analyzer_print.analyze(debug=True)
        print("Metrics:", results_print['metrics'])
        if results_print['graphs']:
             print("Debug graph generated (base64 string).")
             # Optionally save the debug graph
             # with open("debug_print.png", "wb") as f:
             #     f.write(base64.b64decode(results_print['graphs'][0]))

    except ValueError as e:
        print(f"Error processing print image: {e}")
    except FileNotFoundError:
         print(f"Error: Print image file not found at {image_path_print}")


    print("\n--- Analyzing Cursive Sample ---")
    try:
        # Example adjusting some parameters (e.g., slightly larger epsilon if needed)
        analyzer_cursive = VerticalAlignmentAnalyzer(image_path_cursive, is_base64=False)
        # results_cursive = analyzer_cursive.analyze(debug=True, dbscan_epsilon=8.0) # Example override
        results_cursive = analyzer_cursive.analyze(debug=True) # Use dynamic epsilon first
        print("Metrics:", results_cursive['metrics'])
        if results_cursive['graphs']:
             print("Debug graph generated (base64 string).")
             # Optionally save the debug graph
             # with open("debug_cursive.png", "wb") as f:
             #     f.write(base64.b64decode(results_cursive['graphs'][0]))

    except ValueError as e:
        print(f"Error processing cursive image: {e}")
    except FileNotFoundError:
         print(f"Error: Cursive image file not found at {image_path_cursive}")
