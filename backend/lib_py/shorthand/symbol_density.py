import base64
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, binary_dilation
from skimage import morphology, measure
from skimage.color import label2rgb


class SymbolDensityAnalyzer:
    def __init__(self, image_input, is_base64=False):
        """
        Initialize the analyzer with either a base64 encoded image or image file path.

        Parameters:
            image_input (str): Either base64 encoded image string or image file path
            is_base64 (bool): If True, image_input is treated as base64 string, else as file path
        """
        try:
            if is_base64:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if self.img is None:
                    raise ValueError("Error: Could not decode base64 image")
            else:
                # Use OpenCV for consistent reading
                self.img = cv2.imread(image_input, cv2.IMREAD_COLOR)
                if self.img is None:
                    raise ValueError(f"Error: Could not read image at {image_input}")

            # Convert to grayscale
            if self.img.ndim == 3:
                self.gray_img_uint8 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            else:
                # Ensure it's uint8 if it was already grayscale
                if np.issubdtype(self.img.dtype, np.floating):
                    print("Warning: Input grayscale image was float, converting to uint8.")
                    self.gray_img_uint8 = (self.img * 255).clip(0, 255).astype(np.uint8)
                else:
                    self.gray_img_uint8 = self.img.astype(np.uint8)

            if self.gray_img_uint8.size == 0:
                raise ValueError("Error: Image is empty after grayscale conversion.")

        except Exception as e:
            raise ValueError(f"Error during image loading/preprocessing: {e}")

    def _get_bounding_box(self, binary_image):
        """Finds the tight bounding box of True pixels."""
        rows = np.any(binary_image, axis=1)
        cols = np.any(binary_image, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None  # No foreground pixels
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def analyze(self, morph_kernel_size=3, debug=False):
        """
        Analyzes the image using adaptive thresholding and morphological operations.

        Parameters:
            morph_kernel_size (int): Size of the square kernel for morphological operations.
            debug (bool): If True, generates visualization plots.

        Returns:
            dict: A dictionary with metrics and graphs in base64 format (if debug=True)
                  Returns None if processing fails (e.g., blank image after processing).
        """
        try:
            # 1. Adaptive Thresholding (Otsu's method)
            # Assumes dark text on light background. If reverse, use cv2.THRESH_BINARY_INV
            otsu_threshold, binary_thresh = cv2.threshold(
                self.gray_img_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            # Convert OpenCV result (0/255) to boolean (False/True)
            binary = binary_thresh > 0

            # 2. Morphological Cleaning
            kernel = morphology.square(morph_kernel_size)
            # Closing: Fill small gaps within strokes, connect nearby components
            cleaned_binary = morphology.binary_closing(binary, kernel)
            # Opening: Remove small noise specks
            cleaned_binary = morphology.binary_opening(cleaned_binary, kernel)
            # Optional: Fill holes within symbols if needed (might merge unintended areas)
            # cleaned_binary = binary_fill_holes(cleaned_binary)

            # 3. Find Bounding Box and Crop the *cleaned* binary image
            bbox = self._get_bounding_box(cleaned_binary)
            if bbox is None:
                print("Warning: No foreground pixels found after cleaning.")
                # Return default/zeroed metrics if needed, or None
                return None  # Indicate failure or empty result

            rmin, rmax, cmin, cmax = bbox
            binary_cropped = cleaned_binary[rmin:rmax + 1, cmin:cmax + 1]

            if binary_cropped.size == 0:
                print("Warning: Cropped binary image is empty.")
                return None

            # --- Calculate Metrics based on the cleaned, cropped image ---

            # 4. Basic Properties
            text_height = binary_cropped.shape[0]
            text_width = binary_cropped.shape[1]
            text_area = float(binary_cropped.size)  # Use float for ratios
            ink_pixels = int(np.sum(binary_cropped))

            if text_area == 0:
                print("Warning: Text area is zero.")
                return None

            # 5. Connected Components
            labeled_cropped, num_components = label(binary_cropped)

            # 6. Density and Size Metrics
            ink_density = ink_pixels / text_area if text_area > 0 else 0.0
            white_space_ratio = 1.0 - ink_density
            avg_component_area = ink_pixels / num_components if num_components > 0 else 0.0
            # Component density (components per pixel) - interpretation depends on cleaning
            component_density = num_components / text_area if text_area > 0 else 0.0

            # 7. Spacing / Compactness Metrics
            # Dilation on the *cropped* image
            dilated_cropped = binary_dilation(binary_cropped, structure=morphology.square(3))
            # How much of the bounding box is covered by slightly expanded ink
            spacing_compactness = np.sum(dilated_cropped) / text_area if text_area > 0 else 0.0

            # Convex Hull related metrics
            hull_area = 0.0
            convex_hull_ratio = 0.0
            # Use regionprops for convex hull area of the largest component or combined components
            props = measure.regionprops(labeled_cropped, intensity_image=binary_cropped.astype(float))
            if props:
                # Option 1: Use the largest component's hull
                # largest_prop = max(props, key=lambda p: p.area)
                # hull_area = largest_prop.convex_area

                # Option 2: Calculate hull of all points (more representative of overall shape)
                # Convert binary_cropped to uint8 for findContours
                binary_cropped_uint8 = (binary_cropped * 255).astype(np.uint8)
                contours, _ = cv2.findContours(binary_cropped_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_points = np.vstack(
                        [cnt for cnt in contours if cnt.shape[0] > 0])  # Make sure contours are not empty
                    if all_points.shape[0] >= 3:  # Need at least 3 points for a hull
                        hull = cv2.convexHull(all_points)
                        hull_area = cv2.contourArea(hull)
                    elif all_points.shape[0] > 0:  # Handle case of 1 or 2 points (line/point)
                        hull_area = ink_pixels  # Area is just the ink pixels

                # Ratio of ink pixels to the area of their convex hull
                convex_hull_ratio = ink_pixels / hull_area if hull_area > 0 else 0.0

            # Aspect ratio of the text bounding box
            aspect_ratio = text_height / text_width if text_width > 0 else 0.0

            # Original density index - use updated metrics
            density_index = (ink_density + spacing_compactness) / 2.0

            # --- Pack results ---
            metrics = {
                'otsu_threshold': float(otsu_threshold),
                'text_height': text_height,
                'text_width': text_width,
                'text_area': int(text_area),  # Keep as int for consistency? or float? Float makes more sense
                'ink_pixels': ink_pixels,
                'num_components': num_components,  # After cleaning
                'avg_component_area': float(avg_component_area),
                'component_density': float(component_density),
                'ink_density': float(ink_density),  # Proportion of ink in bbox
                'white_space_ratio': float(white_space_ratio),
                'spacing_compactness': float(spacing_compactness),  # How 'filled' bbox is by dilated ink
                'convex_hull_area': float(hull_area),  # Area of the overall shape outline
                'convex_hull_ratio': float(convex_hull_ratio),  # How well ink fills the hull
                'aspect_ratio': float(aspect_ratio),  # Height/Width
                'density_index': float(density_index)  # Original combined score
            }

            result = {
                'metrics': metrics,
                'graphs': []
            }

            # --- Debug Visualization ---
            if debug:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                ax = axes.ravel()

                ax[0].imshow(self.gray_img_uint8, cmap='gray')
                ax[0].set_title('Original Grayscale')
                ax[0].axis('off')

                ax[1].imshow(binary, cmap='gray')
                ax[1].set_title(f'Otsu Threshold ({otsu_threshold:.1f})')
                ax[1].axis('off')

                ax[2].imshow(cleaned_binary, cmap='gray')
                ax[2].set_title('Cleaned Binary')
                ax[2].axis('off')

                ax[3].imshow(binary_cropped, cmap='gray')
                ax[3].set_title(f'Cropped Cleaned ({text_area:.0f} px)')
                ax[3].axis('off')

                # Show components on cropped image
                rgb_label_cropped = label2rgb(labeled_cropped, bg_label=0, image=binary_cropped, image_alpha=0.5)
                ax[4].imshow(rgb_label_cropped)
                ax[4].set_title(f'Components: {num_components}')
                ax[4].axis('off')

                # Show Convex Hull (if calculated via contours)
                vis_img = cv2.cvtColor(binary_cropped_uint8, cv2.COLOR_GRAY2BGR)  # Create color version for drawing
                if 'hull' in locals() and hull is not None:
                    cv2.drawContours(vis_img, [hull], 0, (0, 255, 0), 1)  # Draw green hull
                ax[5].imshow(vis_img)
                ax[5].set_title(f'Hull Ratio: {convex_hull_ratio:.3f}')
                ax[5].axis('off')

                plt.tight_layout()
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)  # Close the figure
                result['graphs'].append(plot_base64)

            return result

        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None  # Indicate failure


# --- Example Usage ---
if __name__ == '__main__':

    # --- Option 1: Read from a file path ---
    # <<< Replace this with the actual path to your image file >>>
    image_file_path = "../../atest/shorthand2.png"
    print(f"Attempting to analyze image from path: {image_file_path}")
    try:
        analyzer = SymbolDensityAnalyzer(image_file_path, is_base64=False) # is_base64=False is default

        # Analyze the image
        analysis_result = analyzer.analyze(debug=True) # Keep debug=True to see plots

        if analysis_result:
            print("\n--- Metrics ---")
            for key, value in analysis_result['metrics'].items():
                print(f"{key}: {value:.4f}")

            print("\n--- Graphs ---")
            if analysis_result['graphs']:
                print(f"Generated {len(analysis_result['graphs'])} graph(s).")
                # Instructions for viewing/saving the base64 plot remain the same
                # e.g., save the base64 string to a file or display in Jupyter
                # with open("debug_plot.png", "wb") as f:
                #    f.write(base64.b64decode(analysis_result['graphs'][0]))
                # print("Saved debug plot to debug_plot.png")
            else:
                print("No graphs generated (debug=False).")
        else:
            print("Analysis failed. Check image path and content.")

    except ValueError as e:
        print(f"Error initializing analyzer or reading file: {e}")
    except FileNotFoundError:
        print(f"Error: The file was not found at '{image_file_path}'. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")