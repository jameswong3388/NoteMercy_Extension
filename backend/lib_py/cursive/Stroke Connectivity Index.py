import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_stroke_connectivity_index(image_path, debug=False):
    # 1. Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read the image at: {image_path}")
        return None

    # Save original image for visualization if debugging
    original = img.copy() if debug else None

    # 2. Improved preprocessing
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

    # 4. Compute stroke connectivity metrics
    component_counts = []
    character_counts = []  # Estimate of character count based on width

    for word_img, bbox in words:
        # Count connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            word_img, connectivity=8
        )

        # Filter out very small components that might be noise or diacritical marks
        valid_components = 0
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i][4]  # Area of the component
            if area > word_img.size * 0.005:  # Only count components larger than 0.5% of word area
                valid_components += 1

        # Estimate number of characters based on width and typical character width
        word_width = bbox[2]
        avg_char_width = word_width / 6 if word_width > 50 else word_width / 3  # Rough estimate
        estimated_chars = max(1, round(word_width / avg_char_width))

        component_counts.append(valid_components)
        character_counts.append(estimated_chars)

    # 5. Calculate stroke connectivity metrics
    # Raw average components per word
    avg_components_per_word = np.mean(component_counts)

    # Components per estimated character (normalization)
    components_per_char = sum(component_counts) / sum(character_counts)

    # Connectivity index: lower values indicate higher connectivity
    # (perfectly connected handwriting would have 1 component per word)
    connectivity_index = (avg_components_per_word - 1) / np.mean(character_counts)

    # Connectivity ratio: the ratio of isolated letter/stroke components to words
    # This is the core measure of the "discreteness" of the writing
    connectivity_ratio = sum(component_counts) / len(words)

    # Normalized connectivity score (0-1 scale, where 1 is fully connected)
    # The idea is that max_components would be equal to char count (fully disconnected)
    # while min_components would be 1 per word (fully connected)
    estimated_total_chars = sum(character_counts)
    connectivity_score = 1 - ((sum(component_counts) - len(words)) /
                              (estimated_total_chars - len(words)))
    # Clamp to 0-1 range in case of estimation errors
    connectivity_score = max(0, min(1, connectivity_score))

    results = {
        "avg_components_per_word": avg_components_per_word,
        "components_per_char": components_per_char,
        "connectivity_index": connectivity_index,
        "connectivity_ratio": connectivity_ratio,
        "connectivity_score": connectivity_score,
        "word_count": len(words),
        "estimated_char_count": sum(character_counts)
    }

    # 6. Visualization for debugging
    if debug:
        # Create a visualization of the detected words
        vis_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        for _, (x, y, w, h) in words:
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

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

        plt.subplot(2, 2, 4)
        plt.bar(['Connectivity Score', 'Connectivity Index', 'Components/Char'],
                [connectivity_score, connectivity_index, components_per_char])
        plt.title('Stroke Connectivity Metrics')

        plt.tight_layout()
        plt.savefig(f"{image_path}_analysis.png")
        plt.close()

    print(f"Stroke Connectivity Analysis Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    return results


if __name__ == "__main__":
    # Replace with your actual image file
    image_path = "atest/3.png"
    results = compute_stroke_connectivity_index(image_path, debug=True)
