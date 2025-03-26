import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology
from skimage.color import label2rgb
from scipy.ndimage import convolve, label
import base64
from io import BytesIO
import cv2


class StrokeContinuityAnalyzer:
    def __init__(self, image_input, is_base64=False):
        """
        Initialize the analyzer with either a base64 encoded image or image file path.

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
            self.img = io.imread(image_input)
            if self.img is None or self.img.size == 0:
                raise ValueError(f"Error: Could not read image at {image_input}")

        # Convert to grayscale if image is RGB
        if self.img.ndim == 3:
            self.gray_img = color.rgb2gray(self.img)
        else:
            # If image is grayscale and in uint8 format, normalize to [0, 1]
            if np.issubdtype(self.img.dtype, np.uint8):
                self.gray_img = self.img / 255.0
            else:
                self.gray_img = self.img

    def analyze(self, debug=False):
        """
        Analyzes the image to determine stroke continuity characteristics.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: A dictionary with metrics and graphs in base64 format (if debug=True)
        """
        # Apply binary thresholding and invert
        threshold = 127 / 255.0
        # In MATLAB, imbinarize uses (I > threshold) then the result is inverted,
        # so here we use (gray <= threshold) for the equivalent effect.
        binary = self.gray_img <= threshold

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
        metrics = {
            'num_components': num_components,
            'num_endpoints': num_endpoints,
            'num_branches': num_branches,
            'components_per_word': components_per_word
        }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # Debug visualization if requested
        if debug:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            ax = axes.ravel()

            # Original Image
            ax[0].imshow(self.img, cmap='gray')
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
            
            # Convert plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            result['graphs'].append(plot_base64)

        return result


# === Example Usage ===
if __name__ == '__main__':
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png'
    analyzer = StrokeContinuityAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])
