import base64
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


class DiscreteLetterAnalyzer:
    def __init__(self, image_input, is_base64=False):
        if is_base64:
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if self.img is None:
                raise ValueError("Error: Could not decode base64 image")
        else:
            self.img = cv2.imread(image_input)
            if self.img is None:
                raise ValueError(f"Error: Could not read image at {image_input}")

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def analyze(self, debug=False):
        if self.img is None:
            return {}

        # Convert to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        # Adaptive thresholding with automatic inversion
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
        binary = adaptive_thresh.astype(bool)

        # Morphological opening to remove noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        binary = cleaned.astype(bool)

        # Label connected components
        labeled_image = label(binary)
        regions = regionprops(labeled_image)

        # Calculate area-based threshold using Otsu
        areas = [r.area for r in regions]
        if len(areas) < 2:
            valid_regions = regions
        else:
            try:
                thresh_area = threshold_otsu(np.array(areas))
                valid_regions = [r for r in regions if r.area >= thresh_area]
            except:
                valid_regions = regions

        # Calculate metrics
        metrics = {
            'num_components': len(valid_regions),
            'total_components': len(regions),
            'avg_area': 0,
            'std_area': 0,
            'avg_aspect_ratio': 0,
            'std_aspect_ratio': 0,
            'avg_solidity': 0
        }

        if valid_regions:
            areas_valid = [r.area for r in valid_regions]
            aspect_ratios = []
            solidities = []

            for r in valid_regions:
                if r.major_axis_length > 0:
                    aspect_ratios.append(r.minor_axis_length / r.major_axis_length)
                else:
                    aspect_ratios.append(0)
                solidities.append(r.solidity if r.solidity else 0)

            metrics.update({
                'avg_area': np.mean(areas_valid),
                'std_area': np.std(areas_valid),
                'avg_aspect_ratio': np.mean(aspect_ratios),
                'std_aspect_ratio': np.std(aspect_ratios),
                'avg_solidity': np.mean(solidities)
            })

        result = {'metrics': metrics, 'graphs': [], 'preprocessed_image': ''}

        # Debug visualization
        if debug:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            titles = [
                ('Original Image', self.img, None),
                ('Cleaned Binary', binary, 'gray'),
                ('Component Overlay', label2rgb(labeled_image, image=self.img, bg_label=0), None),
                ('Valid Components', self.img, None)
            ]

            for ax, (title, data, cmap) in zip(axs.flat, titles):
                ax.imshow(data, cmap=cmap)
                ax.set_title(title)
                ax.axis('off')
                if title == 'Valid Components':
                    for r in valid_regions:
                        minr, minc, maxr, maxc = r.bbox
                        ax.add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                   edgecolor='lime', facecolor='none', lw=2))
                    ax.set_title(f'Valid Components: {len(valid_regions)}')

            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            result['graphs'].append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            plt.show()
            plt.close()

            # Preprocessed image base64
            _, preprocessed_image_buffer = cv2.imencode('.png', cleaned.astype(np.uint8) * 255)
            result['preprocessed_image'] = base64.b64encode(preprocessed_image_buffer).decode('utf-8')

        return result


# Example usage
if __name__ == '__main__':
    analyzer = DiscreteLetterAnalyzer("../../atest/4.png")
    results = analyzer.analyze(debug=True)
    print(results['metrics'])
    metrics = results['metrics']