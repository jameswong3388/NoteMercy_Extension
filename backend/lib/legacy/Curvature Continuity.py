import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def compute_stroke_curvature_continuity(image_path, debug=False):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return {}

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find external contours (focus on external stroke shape)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    segment_lengths = []
    all_polys = []

    # Process each contour
    for cnt in contours:
        # Skip very small contours that might be noise
        if cv2.contourArea(cnt) < 50:
            continue

        # Approximate contour with tolerance proportional to contour perimeter
        epsilon = 0.01 * cv2.arcLength(cnt, closed=True)
        poly = cv2.approxPolyDP(cnt, epsilon, closed=True)
        all_polys.append(poly)

        # Calculate lengths of each segment in the polyline
        for i in range(len(poly)):
            x1, y1 = poly[i][0]
            x2, y2 = poly[(i + 1) % len(poly)][0]  # wrap around closed contour
            seg_len = math.hypot(x2 - x1, y2 - y1)
            segment_lengths.append(seg_len)

    if not segment_lengths:
        return {"avg_normalized_segment_length": 0, "segment_count": 0}

    # Normalize lengths by the image height for invariance
    H = binary.shape[0]
    normalized_segment_lengths = [length / H for length in segment_lengths]

    # Calculate statistics
    avg_normalized_segment_length = np.mean(normalized_segment_lengths)
    median_normalized_segment_length = np.median(normalized_segment_lengths)

    results = dict(avg_normalized_segment_length=avg_normalized_segment_length,
                   median_normalized_segment_length=median_normalized_segment_length,
                   segment_count=len(segment_lengths), total_contours=len(contours))

    # Visualize the results if debug is True
    if debug:
        # Draw original image with contours and simplified polygon
        debug_img = img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        cv2.drawContours(debug_img, all_polys, -1, (0, 0, 255), 2)

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(2, 2, 2)
        plt.imshow(binary, cmap='gray')
        plt.title("Binary Image")

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Contours (green) and Simplified Polygon (red)")

        plt.subplot(2, 2, 4)
        plt.hist(normalized_segment_lengths, bins=20)
        plt.axvline(avg_normalized_segment_length, color='r', linestyle='dashed', linewidth=2)
        plt.title(f"Segment Length Distribution\nAvg: {avg_normalized_segment_length:.3f}")

        plt.tight_layout()
        plt.show()

        print(f"Average normalized segment length: {avg_normalized_segment_length:.3f}")
        print(f"Median normalized segment length: {median_normalized_segment_length:.3f}")
        print(f"Number of segments: {len(segment_lengths)}")
        print(f"Number of contours: {len(contours)}")

    return results


if __name__ == "__main__":
    # Replace with your actual image file
    image_path = "atest/5.png"
    curviness_results = compute_stroke_curvature_continuity(image_path, debug=True)
    print(curviness_results)
