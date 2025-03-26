import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, feature, measure
from scipy.ndimage import distance_transform_edt, convolve
from math import pi


def detect_endpoints(skel):
    """
    Detect endpoints in a binary skeleton.
    An endpoint is defined as a pixel having exactly one neighbor.
    """
    # Define a 3x3 kernel of ones.
    kernel = np.ones((3, 3), dtype=int)
    # Convolve with the binary skeleton.
    neighbors = convolve(skel.astype(np.int32), kernel, mode='constant', cval=0)
    # For a skeleton pixel, count includes the pixel itself.
    endpoints = (skel == 1) & (neighbors == 2)
    return endpoints


def trace_boundary(comp_mask, start_point):
    """
    Trace a simple boundary (connected pixels) starting from start_point.
    This is a simple greedy algorithm that picks the first unvisited 8-neighbor.
    """
    rows, cols = comp_mask.shape
    path = [start_point]
    visited = set([start_point])
    current = start_point
    # Define 8-connected neighborhood offsets.
    neighbors_offset = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 1),
                        (1, -1), (1, 0), (1, 1)]

    while True:
        found_next = False
        cy, cx = current
        for dy, dx in neighbors_offset:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if comp_mask[ny, nx] and ((ny, nx) not in visited):
                    path.append((ny, nx))
                    visited.add((ny, nx))
                    current = (ny, nx)
                    found_next = True
                    break
        if not found_next:
            break
    return np.array(path)


def compute_artistic_consistency(image_path, debug=False):
    # Read and pre-process image
    try:
        img = io.imread(image_path)
    except Exception as e:
        print(f"Error: Could not read image at {image_path}")
        return {}

    # Convert to grayscale if the image is RGB
    if img.ndim == 3:
        if img.shape[2] == 4:
            # Convert RGBA to RGB
            img = color.rgba2rgb(img)
        gray = color.rgb2gray(img)
    else:
        gray = img.astype(np.float64) / 255.0  # Ensure it's in floating point format

    # Adaptive thresholding (using a local threshold)
    # Adjust block_size and offset as needed for robustness.
    block_size = 35  # must be odd
    offset = 0
    local_thresh = filters.threshold_local(gray, block_size, offset=offset)
    binary = gray > local_thresh
    # Complement the binary image (make text white on black background)
    binary = np.invert(binary)

    # Apply morphological filtering to reduce noise and close gaps
    selem = morphology.disk(1)
    binary = morphology.opening(binary, selem)
    binary = morphology.closing(binary, selem)

    # Stroke Width Measurement (Pressure Consistency)
    # Compute skeleton and distance transform.
    skel = morphology.skeletonize(binary)
    D = distance_transform_edt(~binary)

    # Extract stroke widths at skeleton points.
    widths = D[skel]
    widths = widths[widths > 0]  # remove zeros if any

    if widths.size == 0:
        return {
            'pressure_consistency': 0,
            'transition_smoothness': 0,
            'serif_consistency': 0,
            'overall_artistic_score': 0
        }

    width_std = np.std(widths)
    mean_width = np.mean(widths)
    pressure_consistency = 1 - min(width_std / mean_width, 1)

    # Transition Smoothness via Curvature Analysis
    # Label connected components in the skeleton.
    label_img = measure.label(skel, connectivity=2)
    curvature_variances = []

    for region in measure.regionprops(label_img):
        # Create mask for the current stroke component.
        comp_mask = label_img == region.label

        # Identify endpoints in this component.
        comp_endpoints_mask = detect_endpoints(comp_mask)
        endpoints_coords = np.argwhere(comp_endpoints_mask)

        # Trace boundary starting from one endpoint if available.
        if endpoints_coords.shape[0] > 0:
            start_point = tuple(endpoints_coords[0])
            boundary = trace_boundary(comp_mask, start_point)
            # If boundary trace fails or is too short, fallback to unsorted component pixels.
            if boundary.shape[0] < 5:
                boundary = np.array(np.argwhere(comp_mask))
        else:
            boundary = np.array(np.argwhere(comp_mask))

        # Compute directional angles along the traced boundary.
        boundary = boundary.astype(np.float64)
        diff = np.diff(boundary, axis=0)
        angles = np.arctan2(diff[:, 0], diff[:, 1])

        if len(angles) > 1:
            angle_diffs = np.abs(np.diff(angles))
            # Wrap differences so that maximum difference is pi.
            angle_diffs[angle_diffs > np.pi] = 2 * np.pi - angle_diffs[angle_diffs > np.pi]
            curvature_var = np.std(angle_diffs)
            curvature_variances.append(curvature_var)

    if curvature_variances:
        avg_curvature_var = np.mean(curvature_variances)
        transition_smoothness = 1 - min(avg_curvature_var / pi, 1)
    else:
        transition_smoothness = 0.5  # default value

    # Serif Detection using Local Edge Density
    endpoints_mask = detect_endpoints(skel)
    ep_coords = np.argwhere(endpoints_mask)
    serif_scores = []
    window_size = 7
    rows, cols = binary.shape

    for (y, x) in ep_coords:
        y1 = max(0, y - window_size)
        y2 = min(rows, y + window_size + 1)
        x1 = max(0, x - window_size)
        x2 = min(cols, x + window_size + 1)

        local_window = binary[y1:y2, x1:x2]
        # Use Canny edge detection to highlight serif-like features.
        # The sigma parameter can be tuned.
        edges_local = feature.canny(local_window.astype(float), sigma=1)
        density = np.sum(edges_local) / edges_local.size
        serif_scores.append(density)

    if serif_scores and (np.mean(serif_scores) > 0):
        serif_scores = np.array(serif_scores)
        serif_consistency = 1 - min(np.std(serif_scores) / np.mean(serif_scores), 1)
    else:
        serif_consistency = 0.5

    # Combine Metrics to Compute Overall Artistic Consistency Score
    overall_artistic_score = (0.4 * pressure_consistency +
                              0.4 * transition_smoothness +
                              0.2 * serif_consistency)

    results = {
        'pressure_consistency': pressure_consistency,
        'transition_smoothness': transition_smoothness,
        'serif_consistency': serif_consistency,
        'overall_artistic_score': overall_artistic_score
    }

    # Debug Visualization (if requested)
    if debug:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        ax = axes.ravel()

        # Original Image
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        # Binary Image after Adaptive Thresholding and Cleaning
        ax[1].imshow(binary, cmap='gray')
        ax[1].set_title('Binary Image')
        ax[1].axis('off')

        # Skeleton with Endpoints
        ax[2].imshow(binary, cmap='gray')
        skel_y, skel_x = np.nonzero(skel)
        ax[2].scatter(skel_x, skel_y, s=1, c='g')
        ep_y, ep_x = np.nonzero(endpoints_mask)
        ax[2].scatter(ep_x, ep_y, s=20, c='b')
        ax[2].set_title('Skeleton and Endpoints')
        ax[2].axis('off')

        # Heat Map of Stroke Width (Distance Transform Values) along the skeleton
        heatmap = np.zeros_like(binary, dtype=float)
        sy, sx = np.nonzero(skel)
        heatmap[sy, sx] = D[sy, sx]
        im4 = ax[3].imshow(heatmap, cmap='jet')
        ax[3].set_title('Stroke Width Map')
        ax[3].axis('off')
        fig.colorbar(im4, ax=ax[3])

        # Histogram of Stroke Widths
        ax[4].hist(widths, bins=20, color='gray', edgecolor='black')
        ax[4].axvline(mean_width, color='r', linestyle='--', linewidth=2)
        ax[4].set_title(f'Stroke Width Distribution\nConsistency: {pressure_consistency:.2f}')

        # Feature Scores Display
        ax[5].axis('off')
        ax[5].text(0.1, 0.9, f'Pressure: {pressure_consistency:.3f}', fontsize=10)
        ax[5].text(0.1, 0.75, f'Transition: {transition_smoothness:.3f}', fontsize=10)
        ax[5].text(0.1, 0.6, f'Serif: {serif_consistency:.3f}', fontsize=10)
        ax[5].text(0.1, 0.45, f'Overall: {overall_artistic_score:.3f}', fontsize=12, fontweight='bold')
        ax[5].set_title('Feature Scores')

        plt.tight_layout()
        plt.show()

        # Print debug information to console
        print(f'Pressure Consistency: {pressure_consistency:.3f}')
        print(f'Transition Smoothness: {transition_smoothness:.3f}')
        print(f'Serif Consistency: {serif_consistency:.3f}')
        print(f'Overall Artistic Score: {overall_artistic_score:.3f}')

    return results


# === Example Usage ===
if __name__ == '__main__':
    # Replace the image path with your actual file path.
    image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/2.png'
    artistic_results = compute_artistic_consistency(image_path, debug=True)
    print(artistic_results)
