import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology
from skimage.color import label2rgb
from scipy.ndimage import convolve, label


class StrokeContinuityAnalyzer:
    def __init__(self, debug=False):
        """
        Initialize the analyzer.

        Parameters:
            debug (bool): If True, show debug visualizations.
        """
        self.debug = debug

    def compute_stroke_continuity(self, image_path):
        """
        Computes stroke continuity metrics for the given image.

        Parameters:
            image_path (str): Path to the input image.

        Returns:
            dict: A dictionary with keys 'num_components', 'num_endpoints',
                  'num_branches', and 'components_per_word'.
        """
        # Read and preprocess the image
        img = io.imread(image_path)
        if img is None or img.size == 0:
            print(f"Error: Could not read image at {image_path}")
            return {}

        # Convert to grayscale if image is RGB
        if img.ndim == 3:
            gray = color.rgb2gray(img)
        else:
            # If image is grayscale and in uint8 format, normalize to [0, 1]
            if np.issubdtype(img.dtype, np.uint8):
                gray = img / 255.0
            else:
                gray = img

        # Apply binary thresholding and invert
        threshold = 127 / 255.0
        # In MATLAB, imbinarize uses (I > threshold) then the result is inverted,
        # so here we use (gray <= threshold) for the equivalent effect.
        binary = gray <= threshold

        # Get connected components
        # Using SciPy's label function; background is 0.
        labeled, num_components = label(binary)

        # Skeletonize the image (equivalent to MATLAB's bwmorph(binary, 'thin', Inf))
        skel = morphology.skeletonize(binary)

        # Create a 3x3 kernel for neighbor counting (exclude center pixel)
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        # Convolve to count neighbors (convert boolean skeleton to uint8)
        neighbor_count = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)

        # Endpoints: pixels in the skeleton that have exactly one neighbor.
        endpoints = (skel == True) & (neighbor_count == 1)
        # Branch points: pixels in the skeleton that have three or more neighbors.
        branchpoints = (skel == True) & (neighbor_count >= 3)

        num_endpoints = int(np.sum(endpoints))
        num_branches = int(np.sum(branchpoints))

        # In this context, components per word is the number of connected components.
        components_per_word = num_components

        # Pack results into a dictionary
        results = {
            'num_components': num_components,
            'num_endpoints': num_endpoints,
            'num_branches': num_branches,
            'components_per_word': components_per_word
        }

        # Debug visualization if requested
        if self.debug:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            ax = axes.ravel()

            # Original Image
            ax[0].imshow(img, cmap='gray')
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            # Binary Image
            ax[1].imshow(binary, cmap='gray')
            ax[1].set_title('Binary Image')
            ax[1].axis('off')

            # Skeleton with endpoints and branch points overlay
            ax[2].imshow(skel, cmap='gray')
            # Plot endpoints in green
            y_end, x_end = np.where(endpoints)
            ax[2].plot(x_end, y_end, 'go', markersize=6, linewidth=2)
            # Plot branch points in red
            y_branch, x_branch = np.where(branchpoints)
            ax[2].plot(x_branch, y_branch, 'ro', markersize=6, linewidth=2)
            ax[2].set_title('Skeleton with Endpoints (green) and Branch Points (red)')
            ax[2].axis('off')

            # Connected Components visualization using label2rgb
            rgb_label = label2rgb(labeled, bg_label=0)
            ax[3].imshow(rgb_label)
            ax[3].set_title(f'Connected Components ({num_components})')
            ax[3].axis('off')

            plt.tight_layout()
            plt.show()

            # Print debug information to console
            print(f"Number of connected components: {num_components}")
            print(f"Number of endpoints: {num_endpoints}")
            print(f"Number of branch points: {num_branches}")

        return results


# === Example Usage ===
if __name__ == '__main__':
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png'
    analyzer = StrokeContinuityAnalyzer(debug=True)
    continuity_results = analyzer.compute_stroke_continuity(image_path)
    print(continuity_results)
