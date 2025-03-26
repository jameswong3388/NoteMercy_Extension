import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_pen_pressure_consistency(image_path, debug=False):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return {}

    # Convert to grayscale if image is in color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply binary thresholding and invert (threshold value of 127, max value 255)
    # cv2.THRESH_BINARY_INV makes the foreground (pen stroke) white and background black.
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Calculate distance transform
    # This computes, for each non-zero (foreground) pixel, the distance to the nearest zero (background) pixel.
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Get stroke thickness measurements: only consider pixels where binary is non-zero
    thickness_values = dist_transform[binary > 0]

    # If no stroke pixels were detected, return zeros for all measures
    if thickness_values.size == 0:
        return {
            'mean_thickness': 0,
            'thickness_std': 0,
            'thickness_variance': 0,
            'coefficient_of_variation': 0
        }

    # Calculate statistical measures
    mean_thickness = np.mean(thickness_values)
    thickness_std = np.std(thickness_values)
    thickness_variance = np.var(thickness_values)
    coefficient_of_variation = thickness_std / mean_thickness

    results = {
        'mean_thickness': mean_thickness,
        'thickness_std': thickness_std,
        'thickness_variance': thickness_variance,
        'coefficient_of_variation': coefficient_of_variation
    }

    # Debug visualization if requested
    if debug:
        plt.figure("Pen Pressure Analysis", figsize=(10, 8))

        # Original Image
        plt.subplot(2, 2, 1)
        # Convert BGR to RGB for correct display if the image is color
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:
            plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        # Binary Image
        plt.subplot(2, 2, 2)
        plt.imshow(binary, cmap='gray')
        plt.title("Binary Image")
        plt.axis('off')

        # Distance Transform Visualization
        plt.subplot(2, 2, 3)
        plt.imshow(dist_transform, cmap='jet')
        plt.colorbar()
        plt.title("Distance Transform (Stroke Thickness)")
        plt.axis('off')

        # Histogram of thickness values
        plt.subplot(2, 2, 4)
        plt.hist(thickness_values, bins=30)
        plt.axvline(mean_thickness, color='r', linestyle='--', linewidth=2)
        plt.title(f"Thickness Distribution\nMean: {mean_thickness:.2f}, CoV: {coefficient_of_variation:.2f}")
        plt.xlabel("Thickness")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

        # Print debug information
        print(f"Mean stroke thickness: {mean_thickness:.3f}")
        print(f"Thickness standard deviation: {thickness_std:.3f}")
        print(f"Coefficient of variation: {coefficient_of_variation:.3f}")

    return results

# === Example Usage ===
if __name__ == '__main__':
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png'
    pressure_results = compute_pen_pressure_consistency(image_path, debug=True)
    print(pressure_results)
