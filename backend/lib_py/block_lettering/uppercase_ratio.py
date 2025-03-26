import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


def calculate_uppercase_ratio(image_path, debug=False):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return {}

    # Convert to grayscale if image is in color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply binary thresholding and invert
    # MATLAB threshold is 127/255, equivalent to threshold value 127 in 0-255 scale
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)  # Inversion to match typical document analysis

    # Convert binary image to boolean for regionprops
    binary_bool = binary > 0

    # Find connected components (potential characters)
    labeled_img = label(binary_bool)
    props = regionprops(labeled_img)

    if len(props) == 0:
        return {
            'uppercase_ratio': 0,
            'character_count': 0
        }

    # Filter out noise by area: keep components larger than 10% of the median area
    areas = [prop.area for prop in props]
    median_area = np.median(areas)
    filtered_props = [prop for prop in props if prop.area > (median_area * 0.1)]

    if len(filtered_props) == 0:
        return {
            'uppercase_ratio': 0,
            'character_count': 0
        }

    # Extract bounding box heights
    # In skimage, bbox = (min_row, min_col, max_row, max_col)
    heights = [prop.bbox[2] - prop.bbox[0] for prop in filtered_props]
    median_height = np.median(heights)
    normalized_heights = [h / median_height for h in heights]

    # Calculate total image height (for reference)
    img_height = binary.shape[0]

    # Determine uppercase/lowercase classification threshold
    uppercase_threshold = 0.8  # characters with normalized height >= 0.8 considered uppercase-like
    uppercase_count = sum(1 for nh in normalized_heights if nh >= uppercase_threshold)
    total_chars = len(normalized_heights)
    uppercase_ratio = uppercase_count / total_chars if total_chars > 0 else 0

    # Calculate extents (ratio of component pixels to bounding box pixels)
    extents = [prop.extent for prop in filtered_props]
    median_extent = np.median(extents)

    results = {
        'uppercase_ratio': uppercase_ratio,
        'character_count': total_chars,
        'median_height_ratio': median_height / img_height,
        'median_extent': median_extent
    }

    # Debug visualization if requested
    if debug:
        # Prepare figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Uppercase Ratio Analysis", fontsize=16)

        # Original Image (convert BGR to RGB for correct display)
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[0, 0].imshow(img_rgb)
        else:
            axs[0, 0].imshow(img, cmap='gray')
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')

        # Binary Image
        axs[0, 1].imshow(binary, cmap='gray')
        axs[0, 1].set_title('Binary Image')
        axs[0, 1].axis('off')

        # Visualize character bounding boxes on original image
        axs[1, 0].imshow(img_rgb if len(img.shape) == 3 else img, cmap='gray')
        for idx, prop in enumerate(filtered_props):
            minr, minc, maxr, maxc = prop.bbox
            width = maxc - minc
            height = maxr - minr
            # Determine color: red for uppercase, blue for lowercase
            color = 'r' if normalized_heights[idx] >= uppercase_threshold else 'b'
            rect = plt.Rectangle((minc, minr), width, height, edgecolor=color, facecolor='none', linewidth=2)
            axs[1, 0].add_patch(rect)
        axs[1, 0].set_title('Character Classification\n(Red=Uppercase, Blue=Lowercase)')
        axs[1, 0].axis('off')

        # Histogram of normalized heights
        axs[1, 1].hist(normalized_heights, bins=10, edgecolor='black')
        axs[1, 1].axvline(uppercase_threshold, color='red', linestyle='--', linewidth=2)
        axs[1, 1].set_title(f'Height Distribution\nUppercase Ratio: {uppercase_ratio:.2f}')
        axs[1, 1].set_xlabel('Normalized Height')
        axs[1, 1].set_ylabel('Frequency')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # Print debug information
        print(f"Uppercase ratio: {uppercase_ratio:.2f}")
        print(f"Character count: {total_chars}")
        print(f"Median height ratio: {results['median_height_ratio']:.3f}")
        print(f"Median extent: {results['median_extent']:.3f}")

    return results


# === Example usage ===
if __name__ == '__main__':
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/3.png'
    results = calculate_uppercase_ratio(image_path, debug=True)
    print(results)
