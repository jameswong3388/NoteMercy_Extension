import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology
from skimage.color import label2rgb
from scipy.ndimage import label, binary_dilation
import base64
from io import BytesIO
import cv2


class SymbolDensityAnalyzer:
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
        Analyzes the image to determine symbol density characteristics.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: A dictionary with metrics and graphs in base64 format (if debug=True)
        """
        # Apply binary thresholding
        threshold = 127 / 255.0
        binary = self.gray_img <= threshold

        # Get text area (bounding box)
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if np.any(rows) and np.any(cols):
            row_min, row_max = np.where(rows)[0][[0, -1]]
            col_min, col_max = np.where(cols)[0][[0, -1]]
            text_area = (row_max - row_min + 1) * (col_max - col_min + 1)
            # Crop to text area for analysis
            binary_cropped = binary[row_min:row_max+1, col_min:col_max+1]
        else:
            text_area = binary.size
            binary_cropped = binary
        
        # Get connected components
        labeled, num_components = label(binary)
        
        # Calculate ink density (ratio of ink pixels to text area)
        ink_pixels = int(np.sum(binary))
        ink_density = ink_pixels / text_area
        
        # Create a dilated version to measure white space
        dilated = binary_dilation(binary, structure=np.ones((3, 3)))
        
        # Symbol density metrics
        symbol_density = num_components / text_area  # Symbols per pixel area
        avg_symbol_size = ink_pixels / num_components if num_components > 0 else 0
        white_space_ratio = 1.0 - ink_density
        
        # Calculate spacing compactness - how close symbols are to each other
        # Lower value means more space between symbols
        spacing_compactness = np.sum(dilated) / text_area
        
        # Density index (normalized 0-1 score)
        density_index = (ink_density + spacing_compactness) / 2

        # Pack results into a dictionary
        metrics = {
            'num_components': num_components,
            'text_area': int(text_area),
            'ink_pixels': ink_pixels,
            'symbol_density': float(symbol_density),
            'avg_symbol_size': float(avg_symbol_size),
            'ink_density': float(ink_density),
            'white_space_ratio': float(white_space_ratio),
            'spacing_compactness': float(spacing_compactness),
            'density_index': float(density_index)  # Overall density score
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

            # Text Area (cropped)
            ax[2].imshow(binary_cropped, cmap='gray')
            ax[2].set_title(f'Text Area ({text_area} pixels)')
            ax[2].axis('off')

            # Component visualization
            rgb_label = label2rgb(labeled, bg_label=0)
            ax[3].imshow(rgb_label)
            ax[3].set_title(f'Components: {num_components}, Density: {density_index:.3f}')
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