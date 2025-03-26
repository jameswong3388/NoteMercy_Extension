import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
import scipy.interpolate as si


class StrokeSmoothnessAnalyzer:
    def compute_stroke_smoothness(self, image_path, debug=False):
        # Read and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return {}

        # Convert to grayscale if the image is RGB
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Binarize using Otsu's threshold and invert to match expected format
        thresh = threshold_otsu(gray)
        binary = gray > thresh
        binary = np.logical_not(binary)

        # Compute the skeleton of the writing
        skel = skeletonize(binary)

        # Find skeleton points (note: np.nonzero returns (y, x) indices)
        ys, xs = np.nonzero(skel)
        if len(xs) < 3:
            return {'avg_curvature_change': 0,
                    'curvature_variance': 0,
                    'direction_changes': 0,
                    'smoothness_score': 0}

        # Combine coordinates as [x, y] points and sort them along the stroke path
        points = np.column_stack((xs, ys))
        ordered_points = self.sort_skeleton_points(points)
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
            return {}

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

        results = {
            'avg_curvature_change': avg_curvature_change,
            'curvature_variance': curvature_variance,
            'direction_changes': normalized_direction_changes,
            'smoothness_score': smoothness_score
        }

        # Debug visualization if requested
        if debug:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            # Original image (convert BGR to RGB for correct display)
            axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title("Original Image")
            axs[0, 0].axis('off')

            # Skeleton image
            axs[0, 1].imshow(skel, cmap='gray')
            axs[0, 1].set_title("Skeleton")
            axs[0, 1].axis('off')

            # Fitted spline
            axs[1, 0].plot(spline_x, spline_y, 'b-', linewidth=2)
            axs[1, 0].set_aspect('equal')
            axs[1, 0].set_title("Fitted Spline")

            # Curvature changes
            axs[1, 1].plot(dTheta)
            axs[1, 1].set_title("Curvature Changes")

            plt.tight_layout()
            plt.show()

            print(f"Average curvature change: {avg_curvature_change:.3f}")
            print(f"Curvature variance: {curvature_variance:.3f}")
            print(f"Normalized direction changes: {normalized_direction_changes:.3f}")
            print(f"Smoothness score: {smoothness_score:.3f}")

        return results

    @staticmethod
    def sort_skeleton_points(points):
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


# === Example Usage ===
if __name__ == '__main__':
    analyzer = StrokeSmoothnessAnalyzer()
    # Replace the path with the correct image path on your system
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/5.png'
    results = analyzer.compute_stroke_smoothness(image_path, debug=True)
    print(results)
