import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import base64
from io import BytesIO


class DiscreteLetterAnalyzer:
    def __init__(self, image_input, is_base64=False):
        """
        Initialize with either a base64 encoded image or image file path.
        
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
                
        # Convert from BGR (OpenCV default) to RGB
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def analyze(self, debug=False):
        """
        Analyze discrete letter components in the image.

        Parameters:
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: Contains metrics (number of valid letter components, average area, etc.)
                  and base64 encoded visualization graphs if debug=True.
        """
        if self.img is None:
            return {}

        # Convert to grayscale if the image is in color
        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        else:
            gray = self.img

        # Apply binary thresholding using Otsu's method
        thresh = threshold_otsu(gray)
        binary = gray > thresh
        # Invert to match typical text representation (text as white on a dark background)
        binary = np.logical_not(binary)

        # Label connected components
        labeled_image = label(binary)
        regions = regionprops(labeled_image)

        # Calculate the areas of all regions
        areas = [region.area for region in regions]
        if len(areas) == 0:
            median_area = 0
        else:
            median_area = np.median(areas)

        # Filter out noise: consider regions with area > 25% of the median as valid
        valid_regions = [region for region in regions if region.area > 0.25 * median_area]

        num_components = len(valid_regions)
        avg_component_area = np.mean([region.area for region in valid_regions]) if valid_regions else 0

        metrics = {
            'num_letter_components': num_components,
            'avg_component_area': avg_component_area,
            'total_components': len(regions)
        }

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # Debug visualization if requested
        if debug:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # Original Image
            axs[0, 0].imshow(self.img)
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')

            # Binary Image
            axs[0, 1].imshow(binary, cmap='gray')
            axs[0, 1].set_title('Binary Image')
            axs[0, 1].axis('off')

            # Labeled Components (overlay)
            rgb_label = label2rgb(labeled_image, image=self.img, bg_label=0, kind='overlay')
            axs[1, 0].imshow(rgb_label)
            axs[1, 0].set_title('Connected Components')
            axs[1, 0].axis('off')

            # Valid components highlighted
            axs[1, 1].imshow(self.img)
            for region in valid_regions:
                # region.bbox returns (min_row, min_col, max_row, max_col)
                minr, minc, maxr, maxc = region.bbox
                rect_width = maxc - minc
                rect_height = maxr - minr
                rect = plt.Rectangle((minc, minr), rect_width, rect_height,
                                     edgecolor='g', facecolor='none', linewidth=2)
                axs[1, 1].add_patch(rect)
            axs[1, 1].set_title(f'Valid Letters Found: {num_components}')
            axs[1, 1].axis('off')

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
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/4.png'
    analyzer = DiscreteLetterAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)
    print(results['metrics'])
