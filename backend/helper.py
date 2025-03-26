import base64
import io

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

def preprocess_image(image_base64: str):
    """Preprocess image by converting to grayscale and then binary.
    
    Args:
        image_base64: Base64 encoded string of the image
        
    Returns:
        Base64 encoded string of the preprocessed image
    """
    # Convert base64 to image
    img_array = base64_to_image(image_base64)
    
    # Convert to grayscale
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # RGB or RGBA image
        grayscale = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        grayscale = grayscale.astype(np.uint8)
    else:
        # Already grayscale
        grayscale = img_array
    
    # Convert to binary using a threshold
    threshold = 127
    binary = (grayscale > threshold).astype(np.uint8) * 255
    
    # Convert back to base64
    return image_to_base64(binary)
