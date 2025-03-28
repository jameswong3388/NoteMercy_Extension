import base64
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve, label, gaussian_filter
from skimage import io, morphology
from skimage.color import rgb2gray, label2rgb
from skimage.util import img_as_ubyte


class StrokeContinuityAnalyzer:
    def __init__(self, image_input, is_base64=False):
        """
        Initialize the analyzer with either a base64 encoded image or image file path.

        Parameters:
            image_input (str): Either base64 encoded image string or image file path
            is_base64 (bool): If True, image_input is treated as base64 string, else as file path
        """
        if is_base64:
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            # Decode as grayscale directly for simplicity if possible, or color then convert
            img_decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img_decoded is None:
                raise ValueError("Error: Could not decode base64 image")

            if img_decoded.ndim == 3 and img_decoded.shape[2] == 4:  # Handle RGBA
                # Convert RGBA to RGB by blending with white background
                alpha = img_decoded[:, :, 3] / 255.0
                img_rgb = np.zeros_like(img_decoded[:, :, :3])
                for c in range(3):
                    img_rgb[:, :, c] = (1 - alpha) * 255 + alpha * img_decoded[:, :, c]
                self.img = img_rgb.astype(np.uint8)
                self.gray_img_u8 = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
            elif img_decoded.ndim == 3:  # Handle RGB
                self.img = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)  # cv2 reads BGR
                self.gray_img_u8 = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2GRAY)
            else:  # Assume grayscale
                self.img = img_decoded  # Keep original for display if needed
                self.gray_img_u8 = img_decoded

        else:
            # Read image from file path using skimage (often better color handling)
            try:
                self.img = io.imread(image_input)
                if self.img is None or self.img.size == 0:
                    raise ValueError(f"Error: Could not read image at {image_input}")

                # Convert to grayscale uint8 using skimage
                if self.img.ndim == 3:
                    if self.img.shape[2] == 4:  # RGBA
                        self.gray_img_u8 = img_as_ubyte(
                            rgb2gray(self.img[:, :, :3]))  # Basic alpha blend might be better
                    else:  # RGB
                        self.gray_img_u8 = img_as_ubyte(rgb2gray(self.img))
                elif np.issubdtype(self.img.dtype, np.floating):
                    # If grayscale float, convert to uint8
                    self.gray_img_u8 = img_as_ubyte(self.img)
                else:
                    self.gray_img_u8 = self.img  # Assume already grayscale uint8

            except FileNotFoundError:
                raise ValueError(f"Error: Image file not found at {image_input}")
            except Exception as e:
                raise ValueError(f"Error reading image {image_input}: {e}")

        if self.gray_img_u8 is None or self.gray_img_u8.size == 0:
            raise ValueError("Error: Grayscale image could not be obtained.")

    def analyze(self, debug=False):
        """
        Analyzes the image to determine stroke continuity characteristics using improved preprocessing.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: A dictionary with metrics and graphs in base64 format (if debug=True)
        """
        # 1. Noise Reduction (Optional but recommended)
        # Apply a small Gaussian blur. Adjust sigma based on noise level and stroke thickness.
        # Using scipy.ndimage.gaussian_filter on the uint8 image
        blurred_img = gaussian_filter(self.gray_img_u8, sigma=0.7)
        # Alternative using cv2:
        # blurred_img = cv2.GaussianBlur(self.gray_img_u8, (3, 3), 0)

        # 2. Binarization using Otsu's method
        # Assumes dark text on a light background. THRESH_BINARY_INV makes the text white (foreground).
        threshold_value, binary = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Convert result to boolean (True for foreground) for skimage functions
        binary_bool = binary > 0  # or binary.astype(bool)

        # 3. Connected Components
        # label expects integer/boolean input, 0 is background. Our binary_bool works directly.
        labeled, num_components = label(binary_bool)

        # 4. Skeletonization
        # skeletonize expects boolean input where True is the foreground.
        skel = morphology.skeletonize(binary_bool)

        # 5. Feature Point Detection (Endpoints and Branch points)
        # Kernel for neighbor counting (excludes center pixel)
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        # Convolve on the boolean skeleton converted to uint8
        neighbor_count = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)

        # Identify points based on neighbor count on the skeleton
        endpoints = (skel == True) & (neighbor_count == 1)
        branchpoints = (skel == True) & (neighbor_count >= 3)

        num_endpoints = int(np.sum(endpoints))
        num_branches = int(np.sum(branchpoints))

        # Skeleton length (total number of pixels in the skeleton)
        # Might be useful for normalization
        skeleton_length = int(np.sum(skel))

        # Pack results into a dictionary
        metrics = {
            'num_components': num_components,
            'num_endpoints': num_endpoints,
            'num_branches': num_branches,
            'skeleton_length': skeleton_length,
            'otsu_threshold': float(threshold_value)  # Store the threshold found
            # Potentially add normalized features:
            # 'endpoints_per_component': num_endpoints / num_components if num_components > 0 else 0,
            # 'branches_per_component': num_branches / num_components if num_components > 0 else 0,
            # 'endpoints_per_length': num_endpoints / skeleton_length if skeleton_length > 0 else 0,
            # 'branches_per_length': num_branches / skeleton_length if skeleton_length > 0 else 0,
        }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # Debug visualization if requested
        if debug:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Increased size for extra plot
            ax = axes.ravel()

            # Original Image (display the one loaded, could be color or gray)
            ax[0].imshow(self.img, cmap='gray' if self.img.ndim == 2 else None)
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            # Blurred Image
            ax[1].imshow(blurred_img, cmap='gray')
            ax[1].set_title(f'Blurred (Sigma={0.7})')
            ax[1].axis('off')

            # Binary Image (Otsu)
            ax[2].imshow(binary_bool, cmap='gray')
            ax[2].set_title(f'Binary (Otsu Thr: {threshold_value:.1f})')
            ax[2].axis('off')

            # Skeleton with endpoints and branch points overlay
            ax[3].imshow(skel, cmap='gray')
            y_end, x_end = np.where(endpoints)
            ax[3].plot(x_end, y_end, 'go', markersize=5, alpha=0.8, label=f'Endpoints ({num_endpoints})')
            y_branch, x_branch = np.where(branchpoints)
            ax[3].plot(x_branch, y_branch, 'ro', markersize=5, alpha=0.8, label=f'Branches ({num_branches})')
            ax[3].set_title(f'Skeleton (Length: {skeleton_length})')
            ax[3].legend(fontsize='small')
            ax[3].axis('off')

            # Connected Components visualization
            # Use binary_bool for consistency with label input
            rgb_label = label2rgb(labeled, image=~binary_bool, bg_label=0, bg_color=(1, 1, 1), image_alpha=0.5)
            ax[4].imshow(rgb_label)
            ax[4].set_title(f'Connected Components ({num_components})')
            ax[4].axis('off')

            # Clear the last subplot if not used
            ax[5].axis('off')


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
    # Replace with the actual path to your image
    # Use a clear image of a single handwritten word for best results initially
    try:
        # image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png' # Use your path
        image_path = '../../atest/shorthand2.png'  # <--- CHANGE THIS PATH
        print(f"Analyzing image: {image_path}")
        analyzer = StrokeContinuityAnalyzer(image_path, is_base64=False)
        results = analyzer.analyze(debug=True)
        print("\nMetrics:")
        for key, value in results['metrics'].items():
            print(f"  {key}: {value}")

        if results['graphs']:
            print(f"\nGenerated {len(results['graphs'])} debug graph(s).")
            # To save the debug graph (optional):
            # with open("debug_plot.png", "wb") as f:
            #     f.write(base64.b64decode(results['graphs'][0]))
            # print("Debug graph saved as debug_plot.png")

    except ValueError as e:
        print(e)
    except FileNotFoundError:
        print(f"Error: Input image file not found at '{image_path}'. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
