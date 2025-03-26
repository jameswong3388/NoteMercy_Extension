import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_enclosed_loop_ratio(image_path, debug=False):
    # 1. Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read the image at: {image_path}")
        return None

    # Save original image for visualization if debugging
    original = img.copy() if debug else None

    # 2. Preprocessing
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Binarize the image with adaptive thresholding for better handling of varying illumination
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological operations to clean up the binary image
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. Improved word segmentation
    # First, find text line regions using horizontal projection profile
    horizontal_projection = np.sum(binary, axis=1)
    line_threshold = np.max(horizontal_projection) * 0.1  # Adaptive threshold

    lines = []
    in_line = False
    line_start = 0

    for i, proj in enumerate(horizontal_projection):
        if not in_line and proj > line_threshold:
            in_line = True
            line_start = i
        elif in_line and proj <= line_threshold:
            in_line = False
            if i - line_start > 10:  # Minimum line height
                lines.append((line_start, i))

    # Handle case where the last line extends to the bottom of the image
    if in_line:
        lines.append((line_start, len(horizontal_projection) - 1))

    # For each line, segment into words using vertical projection and connected components
    words = []

    for line_start, line_end in lines:
        line_img = binary[line_start:line_end, :]

        # Use connected components with stats to find potential words
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            line_img, connectivity=8
        )

        # Group nearby components as potential words
        word_groups = []
        sorted_components = sorted([(i, stats[i][0]) for i in range(1, num_labels)], key=lambda x: x[1])

        current_group = [sorted_components[0][0]] if len(sorted_components) > 0 else []

        for i in range(1, len(sorted_components)):
            curr_comp = sorted_components[i]
            prev_comp = sorted_components[i - 1]

            # If the horizontal distance between components is small, consider them part of the same word
            if curr_comp[1] - (prev_comp[1] + stats[prev_comp[0]][2]) < stats[prev_comp[0]][2] * 0.8:
                current_group.append(curr_comp[0])
            else:
                if current_group:
                    word_groups.append(current_group)
                current_group = [curr_comp[0]]

        if current_group:
            word_groups.append(current_group)

        # Extract words from groups
        for group in word_groups:
            # Create a mask for the current word
            word_mask = np.zeros_like(line_img)
            for comp_id in group:
                word_mask[labels == comp_id] = 255

            # Get the bounding box of the word
            word_coords = np.column_stack(np.where(word_mask > 0))
            if len(word_coords) == 0:
                continue

            y_min, x_min = np.min(word_coords, axis=0)
            y_max, x_max = np.max(word_coords, axis=0)

            # Add global line offset to y coordinates
            y_min += line_start
            y_max += line_start

            # Extract the word image from the original binary image
            word_img = binary[y_min:y_max + 1, x_min:x_max + 1]

            # Filter out very small regions that might be noise
            if word_img.shape[0] > 10 and word_img.shape[1] > 10:
                words.append((word_img, (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)))

    # If no words are found, exit
    if not words:
        print("No word-like regions found in the image.")
        return None

    # 4. Compute enclosed loop metrics
    total_outer_count = 0
    total_inner_count = 0
    word_loopiness = []

    for word_img, bbox in words:
        # Find contours with hierarchy information to detect loops
        contours, hierarchy = cv2.findContours(
            word_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None or len(hierarchy) == 0:
            continue

        outer_count = inner_count = 0
        for idx, cnt in enumerate(contours):
            # Check contour area to filter out noise
            area = cv2.contourArea(cnt)
            if area < 10:  # Skip very small contours that might be noise
                continue

            # hierarchy[0][idx][3] == -1 indicates an outer contour (no parent)
            if hierarchy[0][idx][3] == -1:
                outer_count += 1
            else:
                inner_count += 1

        # Calculate loopiness for this word
        word_loop_ratio = inner_count / outer_count if outer_count > 0 else 0
        word_loopiness.append(word_loop_ratio)

        total_outer_count += outer_count
        total_inner_count += inner_count

    # 5. Calculate global loopiness metrics
    # Overall loopiness ratio
    global_loopiness = total_inner_count / total_outer_count if total_outer_count > 0 else 0

    # Average loopiness across words
    avg_word_loopiness = np.mean(word_loopiness) if word_loopiness else 0

    # Standard deviation of loopiness (consistency metric)
    std_loopiness = np.std(word_loopiness) if len(word_loopiness) > 1 else 0

    results = {
        "global_loopiness": global_loopiness,
        "avg_word_loopiness": avg_word_loopiness,
        "std_loopiness": std_loopiness,
        "inner_contour_count": total_inner_count,
        "outer_contour_count": total_outer_count,
        "word_count": len(words)
    }

    # 6. Visualization for debugging
    if debug:
        # Create visualizations of the enclosed loops
        vis_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

        # Draw bounding boxes around detected words
        for _, (x, y, w, h) in words:
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Visualization for a sample word to show loops
        if words:
            sample_word_img, sample_bbox = words[len(words) // 2]  # Use middle word as sample
            sample_contours, sample_hierarchy = cv2.findContours(
                sample_word_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )

            sample_vis = cv2.cvtColor(sample_word_img, cv2.COLOR_GRAY2BGR)

            # Draw outer contours in green, inner contours (loops) in red
            for idx, cnt in enumerate(sample_contours):
                if sample_hierarchy[0][idx][3] == -1:  # Outer contour
                    cv2.drawContours(sample_vis, [cnt], 0, (0, 255, 0), 2)
                else:  # Inner contour (loop)
                    cv2.drawContours(sample_vis, [cnt], 0, (0, 0, 255), 2)

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title('Original Image')

        plt.subplot(2, 2, 2)
        plt.imshow(binary, cmap='gray')
        plt.title('Binarized Image')

        plt.subplot(2, 2, 3)
        plt.imshow(vis_img)
        plt.title('Detected Words')

        if words:
            plt.subplot(2, 2, 4)
            plt.imshow(sample_vis)
            plt.title(f'Sample Word: Loops in Red (Loopiness: {word_loopiness[len(words) // 2]:.2f})')
        else:
            plt.subplot(2, 2, 4)
            plt.bar(['Global Loopiness', 'Avg Word Loopiness'],
                    [global_loopiness, avg_word_loopiness])
            plt.title('Loopiness Metrics')

        plt.tight_layout()
        plt.savefig(f"{image_path}_loopiness_analysis.png")
        plt.close()

    print(f"Enclosed Loop Ratio Analysis Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    return results


if __name__ == "__main__":
    # Replace with your actual image file
    image_path = "atest/5.png"
    results = compute_enclosed_loop_ratio(image_path, debug=True)
