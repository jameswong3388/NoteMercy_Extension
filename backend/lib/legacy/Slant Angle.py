import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_slant_angle(image_path, debug=False):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out very small contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    # Calculate slant angle for each contour
    slant_angles = []
    for cnt in contours:
        M = cv2.moments(cnt)
        # Skip components with nearly zero second moment or perfectly symmetric ones
        if M['m00'] == 0 or M['mu02'] < 1e-2:
            continue

        # Calculate skew using central moments
        skew = M['mu11'] / M['mu02']

        # Convert skew to angle in degrees
        angle_rad = math.atan(skew)
        angle_deg = angle_rad * (180.0 / np.pi)

        slant_angles.append(angle_deg)

    # Calculate statistics
    results = {}
    if slant_angles:
        avg_slant = np.mean(slant_angles)
        slant_std = np.std(slant_angles)

        # Convert to angle from vertical (for more intuitive interpretation)
        vertical_slant = 90 - avg_slant if avg_slant <= 90 else avg_slant - 90

        results = {
            'avg_slant': avg_slant,
            'vertical_slant': vertical_slant,
            'slant_std': slant_std,
            'num_components': len(slant_angles)
        }

    # Debug visualization
    if debug and slant_angles:
        # Draw contours and their principal axes
        vis_img = img.copy()
        cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)

        # Plot histogram of slant angles
        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title("Contours")

        plt.subplot(122)
        plt.hist(slant_angles, bins=20, color='blue', alpha=0.7)
        plt.axvline(x=avg_slant, color='r', linestyle='--',
                    label=f'Avg: {avg_slant:.1f}°, StdDev: {slant_std:.1f}°')
        plt.title("Slant Angle Distribution")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Average slant: {avg_slant:.1f}°")
        print(f"Angle from vertical: {vertical_slant:.1f}°")
        print(f"Slant consistency (std): {slant_std:.1f}°")
        print(f"Components analyzed: {len(slant_angles)}")

    return results


if __name__ == "__main__":
    # Replace with your actual image file
    image_path = "atest/3.png"
    curviness_results = compute_slant_angle(image_path, debug=True)
    print(curviness_results)
