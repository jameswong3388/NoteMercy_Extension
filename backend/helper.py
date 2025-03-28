import base64
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def base64_to_image(base64_string: str):
    """Convert base64 string to numpy array image."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode base64 string
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


def image_to_base64(img_array):
    """Convert numpy array image to base64 string."""
    try:
        img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        raise ValueError(f"Error converting image to base64: {str(e)}")


def save_plot_to_base64():
    """Save current matplotlib plot to base64 string."""
    buffered = io.BytesIO()
    plt.savefig(buffered, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buffered.getvalue()).decode()


def preprocess_image(
        image_base64: str,
        target_height: int = None,  # Optional: Normalize height
        blur_ksize: int = 3,  # Kernel size for Gaussian blur (odd number), 0 or 1 to disable
        adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Or cv2.ADAPTIVE_THRESH_MEAN_C
        adaptive_block_size: int = 11,  # Size of the neighborhood area (odd number > 1)
        adaptive_C: int = 5  # Constant subtracted from the mean or weighted sum
) -> str:
    """
    Preprocesses a handwriting image (word level) using improved techniques.

    Steps:
    1. Decode base64.
    2. Convert to grayscale.
    3. Optional: Apply Gaussian blur for noise reduction.
    4. Apply adaptive thresholding for robust binarization.
    5. Invert image (ensure black text on white background).
    6. Optional: Normalize image height.
    7. Encode back to base64.

    Args:
        image_base64: Base64 encoded string of the image.
        target_height: If set, resize image to this height, maintaining aspect ratio.
        blur_ksize: Kernel size for Gaussian Blur. Set to 0 or 1 to disable. Must be odd.
        adaptive_method: cv2.ADAPTIVE_THRESH_GAUSSIAN_C or cv2.ADAPTIVE_THRESH_MEAN_C.
        adaptive_block_size: Size of the pixel neighborhood used to calculate the threshold. Must be odd > 1.
        adaptive_C: Constant subtracted from the calculated threshold.

    Returns:
        Base64 encoded string of the preprocessed (binary) image.
    """
    try:
        # 1. Decode base64 to OpenCV image
        img = base64_to_image(image_base64)

        # 2. Convert to grayscale
        if len(img.shape) == 3 and img.shape[2] >= 3:
            # Convert BGR or BGRA to Grayscale
            # Note: OpenCV reads images as BGR by default
            if img.shape[2] == 4:  # BGRA
                grayscale = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:  # BGR
                grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            # Already grayscale
            grayscale = img
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        # 3. Optional: Noise Reduction
        if blur_ksize > 1 and blur_ksize % 2 == 1:
            blurred = cv2.GaussianBlur(grayscale, (blur_ksize, blur_ksize), 0)
        else:
            blurred = grayscale  # Skip blurring if ksize is invalid or <= 1

        # 4. Adaptive Binarization
        if adaptive_block_size <= 1 or adaptive_block_size % 2 == 0:
            print(f"Warning: adaptive_block_size ({adaptive_block_size}) must be an odd number > 1. Using default 11.")
            adaptive_block_size = 11

        binary = cv2.adaptiveThreshold(
            blurred,
            maxValue=255,
            adaptiveMethod=adaptive_method,
            thresholdType=cv2.THRESH_BINARY,  # Results in white object on black background
            blockSize=adaptive_block_size,
            C=adaptive_C
        )

        # 5. Invert image (to get black text on white background, common convention)
        #    If your feature extractor expects white text on black, skip this step.
        binary_inverted = cv2.bitwise_not(binary)

        processed_image = binary_inverted

        # 6. Optional: Size Normalization (based on height)
        if target_height is not None and target_height > 0:
            h = processed_image.shape[0]
            if h > 0:  # Avoid division by zero for empty images
                scale_factor = target_height / h
                new_w = int(processed_image.shape[1] * scale_factor)
                # Use INTER_LINEAR or INTER_CUBIC for better quality, INTER_NEAREST for speed
                processed_image = cv2.resize(processed_image, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
                # After resizing, re-binarize slightly to ensure clean edges? Sometimes needed.
                # _, processed_image = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY)

        # 7. Convert back to base64
        return image_to_base64(processed_image)

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        # Depending on requirements, either return an empty/error string or re-raise
        # return ""
        raise
