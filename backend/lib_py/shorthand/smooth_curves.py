import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
import scipy.interpolate as si
import base64
from io import BytesIO


class StrokeSmoothnessAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the StrokeSmoothnessAnalyzer with either a base64 encoded image or image path.

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

    @staticmethod
    def _sort_skeleton_points(points):
        """
        Sorts skeleton points along the stroke path.
        Starts with the leftmost point and repeatedly connects to the nearest neighbor.
        """
        if len(points) == 0:
            return points

        # Start with the leftmost point (minimum x-coordinate)
        idx = np.argmin(points[:, 0])
        ordered_points = [points[idx]]
        points = np.delete(points, idx, axis=0)

        # Connect remaining points by nearest neighbor
        while len(points) > 0:
            current = ordered_points[-1]
            distances = np.linalg.norm(points - current, axis=1)
            idx = np.argmin(distances)
            ordered_points.append(points[idx])
            points = np.delete(points, idx, axis=0)

        return np.array(ordered_points)

    def analyze(self, debug=False):
        """
        Analyzes the image to determine stroke smoothness characteristics.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Metrics including curvature information and smoothness score,
                  plus visualization graphs if debug=True.
        """
        # Binarize using Otsu's threshold and invert to match expected format
        thresh = threshold_otsu(self.gray_img)
        binary = self.gray_img > thresh
        binary = np.logical_not(binary)

        # Compute the skeleton of the writing
        skel = skeletonize(binary)

        # Find skeleton points (note: np.nonzero returns (y, x) indices)
        ys, xs = np.nonzero(skel)
        if len(xs) < 3:
            metrics = {
                'avg_curvature_change': 0,
                'curvature_variance': 0,
                'direction_changes': 0,
                'smoothness_score': 0
            }
            return {'metrics': metrics, 'graphs': []}

        # Combine coordinates as [x, y] points and sort them along the stroke path
        points = np.column_stack((xs, ys))
        ordered_points = self._sort_skeleton_points(points)
        x_sorted = ordered_points[:, 0]
        y_sorted = ordered_points[:, 1]

        # Fit a parametric spline through the skeleton points
        try:
            tck, _ = si.splprep([x_sorted, y_sorted], s=0)
            u_fine = np.linspace(0, 1, 100)
            spline = si.splev(u_fine, tck)
            spline_x, spline_y = spline
        except Exception as e:
            print(f"Error during spline fitting: {e}")
            metrics = {
                'avg_curvature_change': 0,
                'curvature_variance': 0,
                'direction_changes': 0,
                'smoothness_score': 0
            }
            return {'metrics': metrics, 'graphs': []}

        # Calculate gradients and direction changes
        dx = np.gradient(spline_x)
        dy = np.gradient(spline_y)
        theta = np.degrees(np.arctan2(dy, dx))
        theta_rad = np.radians(theta)
        theta_unwrapped = np.unwrap(theta_rad)
        dTheta = np.diff(theta_unwrapped)

        # Compute metrics for curvature and direction changes
        avg_curvature_change = np.mean(np.abs(dTheta))
        curvature_variance = np.var(dTheta)
        direction_changes = np.sum(np.abs(np.diff(np.sign(dTheta))) > 0)

        # Normalize by stroke length
        stroke_length = np.sum(np.sqrt(np.diff(spline_x) ** 2 + np.diff(spline_y) ** 2))
        normalized_direction_changes = direction_changes / stroke_length

        # Calculate overall smoothness score (lower is smoother)
        epsilon = 1e-6  # small constant to avoid multiplication by zero
        smoothness_score = (max(avg_curvature_change, epsilon) *
                            max(normalized_direction_changes, epsilon) *
                            max(curvature_variance, epsilon)) ** (1 / 3)
        smoothness_score = min(100, smoothness_score * 10)

        metrics = {
            'avg_curvature_change': avg_curvature_change,
            'curvature_variance': curvature_variance,
            'direction_changes': normalized_direction_changes,
            'smoothness_score': smoothness_score
        }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # Debug visualization if requested
        if debug:
            # Create figure with subplots
            plt.figure("Stroke Smoothness Analysis", figsize=(10, 10))

            # Original image
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            # Skeleton image
            plt.subplot(2, 2, 2)
            plt.imshow(skel, cmap='gray')
            plt.title("Skeleton")
            plt.axis('off')

            # Fitted spline
            plt.subplot(2, 2, 3)
            plt.plot(spline_x, spline_y, 'b-', linewidth=2)
            plt.set_aspect = 'equal'
            plt.title("Fitted Spline")

            # Curvature changes
            plt.subplot(2, 2, 4)
            plt.plot(dTheta)
            plt.title(f"Curvature Changes\nSmoothness Score: {smoothness_score:.2f}")

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
if __name__ == '__main__':
    # Example with file path
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/5.png'
    analyzer = StrokeSmoothnessAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])
