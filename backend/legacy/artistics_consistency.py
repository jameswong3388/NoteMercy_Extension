import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, feature, measure
from scipy.ndimage import distance_transform_edt, convolve
from math import pi
import base64
from io import BytesIO


class CalligraphicAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the CalligraphicAnalyzer with either a base64 encoded image or image path.

        Parameters:
            image_input (str): Either base64 encoded image string or image file path
            is_base64 (bool): If True, image_input is treated as base64 string, else as file path
        """
        if is_base64:
            # Decode base64 image
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            self.img = io.imread(BytesIO(nparr))
            if self.img is None:
                raise ValueError("Error: Could not decode base64 image")
        else:
            # Read image from file path
            self.img = io.imread(image_input)
            if self.img is None:
                raise ValueError(f"Error: Could not read image at {image_input}")

        self.gray = None
        self.binary = None
        self.skel = None
        self.D = None

    @staticmethod
    def _detect_endpoints(skel):
        """
        Detect endpoints in a binary skeleton.
        An endpoint is defined as a pixel having exactly one neighbor.
        """
        kernel = np.ones((3, 3), dtype=int)
        neighbors = convolve(skel.astype(np.int32), kernel, mode='constant', cval=0)
        endpoints = (skel == 1) & (neighbors == 2)
        return endpoints

    @staticmethod
    def _trace_boundary(comp_mask, start_point):
        """
        Trace a simple boundary (connected pixels) starting from start_point.
        This is a simple greedy algorithm that picks the first unvisited 8-neighbor.
        """
        rows, cols = comp_mask.shape
        path = [start_point]
        visited = {start_point}
        current = start_point
        # Define 8-connected neighborhood offsets.
        neighbors_offset = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1), (0, 1),
                            (1, -1), (1, 0), (1, 1)]

        while True:
            found_next = False
            cy, cx = current
            for dy, dx in neighbors_offset:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    if comp_mask[ny, nx] and ((ny, nx) not in visited):
                        path.append((ny, nx))
                        visited.add((ny, nx))
                        current = (ny, nx)
                        found_next = True
                        break
            if not found_next:
                break
        return np.array(path)

    def _load_and_preprocess(self):
        """
        Load image and preprocess it.
        """
        # Convert to grayscale if the image is RGB.
        if self.img.ndim == 3:
            if self.img.shape[2] == 4:
                # Convert RGBA to RGB.
                self.img = color.rgba2rgb(self.img)
            self.gray = color.rgb2gray(self.img)
        else:
            self.gray = self.img.astype(np.float64) / 255.0

        # Adaptive thresholding (local thresholding).
        block_size = 35  # must be odd.
        offset = 0
        local_thresh = filters.threshold_local(self.gray, block_size, offset=offset)
        self.binary = self.gray > local_thresh
        # Complement the binary image.
        self.binary = np.invert(self.binary)

        # Morphological filtering.
        selem = morphology.disk(1)
        self.binary = morphology.opening(self.binary, selem)
        self.binary = morphology.closing(self.binary, selem)

    def analyze(self, debug=False):
        """
        Analyze the image to determine calligraphic characteristics by measuring
        pressure consistency, transition smoothness, and serif consistency.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Metrics including pressure consistency, transition smoothness,
                  serif consistency, and overall artistic score, plus visualization graphs if debug=True.
        """
        self._load_and_preprocess()

        # Skeleton and distance transform.
        self.skel = morphology.skeletonize(self.binary)
        self.D = distance_transform_edt(~self.binary)

        # Stroke Width Measurement (Pressure Consistency).
        widths = self.D[self.skel]
        widths = widths[widths > 0]  # Remove zeros.

        if widths.size == 0:
            metrics = {
                'pressure_consistency': 0,
                'transition_smoothness': 0,
                'serif_consistency': 0,
                'overall_artistic_score': 0
            }
            return {'metrics': metrics, 'graphs': [], 'preprocessed_image': ''}

        width_std = np.std(widths)
        mean_width = np.mean(widths)
        pressure_consistency = 1 - min(width_std / mean_width, 1)

        # Transition Smoothness via Curvature Analysis.
        label_img = measure.label(self.skel, connectivity=2)
        curvature_variances = []

        for region in measure.regionprops(label_img):
            comp_mask = label_img == region.label
            comp_endpoints_mask = self._detect_endpoints(comp_mask)
            endpoints_coords = np.argwhere(comp_endpoints_mask)

            if endpoints_coords.shape[0] > 0:
                start_point = tuple(endpoints_coords[0])
                boundary = self._trace_boundary(comp_mask, start_point)
                if boundary.shape[0] < 5:
                    boundary = np.array(np.argwhere(comp_mask))
            else:
                boundary = np.array(np.argwhere(comp_mask))

            boundary = boundary.astype(np.float64)
            diff = np.diff(boundary, axis=0)
            angles = np.arctan2(diff[:, 0], diff[:, 1])

            if len(angles) > 1:
                angle_diffs = np.abs(np.diff(angles))
                angle_diffs[angle_diffs > np.pi] = 2 * np.pi - angle_diffs[angle_diffs > np.pi]
                curvature_var = np.std(angle_diffs)
                curvature_variances.append(curvature_var)

        if curvature_variances:
            avg_curvature_var = np.mean(curvature_variances)
            transition_smoothness = 1 - min(avg_curvature_var / pi, 1)
        else:
            transition_smoothness = 0.5  # Default value.

        # Serif Detection using Local Edge Density.
        endpoints_mask = self._detect_endpoints(self.skel)
        ep_coords = np.argwhere(endpoints_mask)
        serif_scores = []
        window_size = 7
        rows, cols = self.binary.shape

        for (y, x) in ep_coords:
            y1 = max(0, y - window_size)
            y2 = min(rows, y + window_size + 1)
            x1 = max(0, x - window_size)
            x2 = min(cols, x + window_size + 1)

            local_window = self.binary[y1:y2, x1:x2]
            edges_local = feature.canny(local_window.astype(float), sigma=1)
            density = np.sum(edges_local) / edges_local.size
            serif_scores.append(density)

        if serif_scores and (np.mean(serif_scores) > 0):
            serif_scores = np.array(serif_scores)
            serif_consistency = 1 - min(np.std(serif_scores) / np.mean(serif_scores), 1)
        else:
            serif_consistency = 0.5

        overall_artistic_score = (0.4 * pressure_consistency +
                                  0.4 * transition_smoothness +
                                  0.2 * serif_consistency)

        metrics = {
            'pressure_consistency': pressure_consistency,
            'transition_smoothness': transition_smoothness,
            'serif_consistency': serif_consistency,
            'overall_artistic_score': overall_artistic_score
        }

        result = {
            'metrics': metrics,
            'graphs': [],
            'preprocessed_image': ''
        }

        if debug:
            # Create visualization plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            ax = axes.ravel()

            # Original Image
            ax[0].imshow(self.img, cmap='gray')
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            # Binary Image
            ax[1].imshow(self.binary, cmap='gray')
            ax[1].set_title('Binary Image')
            ax[1].axis('off')

            # Skeleton with Endpoints
            ax[2].imshow(self.binary, cmap='gray')
            skel_y, skel_x = np.nonzero(self.skel)
            ax[2].scatter(skel_x, skel_y, s=1, c='g')
            ep_y, ep_x = np.nonzero(endpoints_mask)
            ax[2].scatter(ep_x, ep_y, s=20, c='b')
            ax[2].set_title('Skeleton and Endpoints')
            ax[2].axis('off')

            # Heat Map of Stroke Width
            heatmap = np.zeros_like(self.binary, dtype=float)
            sy, sx = np.nonzero(self.skel)
            heatmap[sy, sx] = self.D[sy, sx]
            im4 = ax[3].imshow(heatmap, cmap='jet')
            ax[3].set_title('Stroke Width Map')
            ax[3].axis('off')
            fig.colorbar(im4, ax=ax[3])

            # Histogram of Stroke Widths
            ax[4].hist(widths, bins=20, color='gray', edgecolor='black')
            ax[4].axvline(mean_width, color='r', linestyle='--', linewidth=2)
            ax[4].set_title(f'Stroke Width Distribution\nConsistency: {metrics["pressure_consistency"]:.2f}')

            # Feature Scores Display
            ax[5].axis('off')
            ax[5].text(0.1, 0.9, f'Pressure: {metrics["pressure_consistency"]:.3f}', fontsize=10)
            ax[5].text(0.1, 0.75, f'Transition: {metrics["transition_smoothness"]:.3f}', fontsize=10)
            ax[5].text(0.1, 0.6, f'Serif: {metrics["serif_consistency"]:.3f}', fontsize=10)
            ax[5].text(0.1, 0.45, f'Overall: {metrics["overall_artistic_score"]:.3f}',
                       fontsize=12, fontweight='bold')
            ax[5].set_title('Feature Scores')

            plt.tight_layout()

            # Convert plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            result['graphs'].append(plot_base64)

            # preprocessed image to base64
            buf2 = BytesIO()
            plt.imshow(self.binary, cmap='gray')
            plt.axis('off')
            plt.savefig(buf2, format='png', bbox_inches='tight')
            plt.close()
            buf2.seek(0)
            preprocessed_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
            result['preprocessed_image'] = preprocessed_base64

        return result


if __name__ == '__main__':
    # Example with file path
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/2.png'
    analyzer = CalligraphicAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results)