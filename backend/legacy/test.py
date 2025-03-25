import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def compute_vertical_stroke_proportion(image_path, debug=False):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Find connected components (letters or parts of letters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Extract bounding box information for each component
    boxes = []
    for i in range(1, num_labels):  # Skip background (label 0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # Filter out noise (very small components)
        if h > 5 and w > 5:  
            boxes.append((x, y, w, h))
    
    # Get heights of all connected components
    heights = [h for (x, y, w, h) in boxes]
    
    results = {}
    if len(heights) > 0:
        heights.sort()
        median_height = np.median(heights)   # approximate x-height (middle zone)
        max_height = max(heights)           # height of tallest stroke (ascender or capital)
        ascender_ratio = max_height / median_height if median_height > 0 else 0
        
        results = {
            "median_height": float(median_height),
            "max_height": float(max_height),
            "ascender_ratio": float(ascender_ratio)
        }
        
        print(f"Ascender-to-xHeight ratio: {ascender_ratio:.2f}")
    
    # Debug visualization
    if debug and len(boxes) > 0:
        # Draw bounding boxes on the original image
        img_with_boxes = img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Create a figure with subplots
        plt.figure(figsize=(15, 10))
        
        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(132)
        plt.title("Binary Image")
        plt.imshow(binary, cmap='gray')
        
        plt.subplot(133)
        plt.title(f"Bounding Boxes (Ascender Ratio: {ascender_ratio:.2f})")
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        
        plt.tight_layout()
        plt.show()
    
    return results

if __name__ == "__main__":
    # Replace with your actual image file
    image_path = "../atest/3.png"
    curviness_results = compute_vertical_stroke_proportion(image_path, debug=True)
    print(curviness_results)
