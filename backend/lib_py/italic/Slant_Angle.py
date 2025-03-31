import base64
from io import BytesIO

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use a non-interactive backend suitable for saving figures
matplotlib.use('Agg')

class SlantAngleAnalyzer:
    def __init__(self, image_input, is_base64=False):
        """
        Initializes the analyzer with an image.

        Args:
            image_input (str): Path to the image file or base64 encoded string.
            is_base64 (bool): True if image_input is a base64 string, False if it's a path.
        """
        try:
            if is_base64:
                img_data = base64.b64decode(image_input)
                nparr = np.frombuffer(img_data, np.uint8)
                self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if self.img is None:
                    raise ValueError("Error: Could not decode base64 image")
            else:
                self.img = cv2.imread(image_input)
                if self.img is None:
                    raise ValueError(f"Error: Could not read image at {image_input}")

            if len(self.img.shape) == 3 and self.img.shape[2] == 3:
                self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            elif len(self.img.shape) == 2:
                # Already grayscale, but ensure it's 3-channel for visualization consistency
                self.gray_img = self.img.copy()
                self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f"Error: Unexpected image shape {self.img.shape}")


        except Exception as e:
            print(f"Error during image loading/preprocessing: {e}")
            raise

        self.contours = None
        self.ellipses = None  # Store fitted ellipses
        self.slant_angles = []
        self.vertical_slants = []

    def _preprocess_image(self):
        """Applies thresholding and finds contours."""
        # Apply adaptive thresholding or Otsu's depending on image characteristics
        # Using Otsu's thresholding as in the original code
        _, binary = cv2.threshold(
            self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Optional morphological operations to clean up noise or connect components
        # kernel = np.ones((2,2),np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and aspect ratio if needed
        min_contour_area = 50  # As before
        self.contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        # print(f"Found {len(self.contours)} suitable contours.")


    def _calculate_slant_angles(self):
        """Calculates slant angles by fitting ellipses to contours."""
        angles = []
        ellipses = []
        valid_contours = []  # Keep track of contours for which ellipse fitting succeeded
        for cnt in self.contours:
            if len(cnt) >= 5:  # fitEllipse requires at least 5 points
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    angles.append(ellipse[2])  # Angle in degrees (0-180)
                    ellipses.append(ellipse)
                    valid_contours.append(cnt)
                except cv2.error as e:
                    # Sometimes fitEllipse can fail on degenerate contours
                    # print(f"Warning: Could not fit ellipse for a contour: {e}")
                    pass  # Skip this contour
            # else:
            # print(f"Warning: Skipping contour with only {len(cnt)} points.")

        # Update self.contours to only include those where fitting succeeded
        self.contours = valid_contours
        return angles, ellipses

    def _generate_debug_plots(self, metrics):
        """Generates a multi-panel figure visualizing the analysis."""
        if not self.slant_angles or not self.contours:
            print("No valid slant angles or contours found to generate debug plots.")
            return []  # Return empty list if no data

        vis_img = self.img.copy()

        # Draw contours (green)
        cv2.drawContours(vis_img, self.contours, -1, (0, 255, 0), 1)  # Thinner lines

        # Draw fitted ellipses (red)
        if self.ellipses:
            for ellipse in self.ellipses:
                # Ensure ellipse parameters are valid tuples/numbers for cv2.ellipse
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))  # Halve axes lengths
                angle = ellipse[2]
                # Draw ellipse outline
                cv2.ellipse(vis_img, center, axes, angle, 0, 360, (0, 0, 255), 1)  # Red ellipse

        # Create plot figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Increased figure size
        fig.suptitle('Slant Angle Analysis Results', fontsize=16)

        # --- Plot 1: Image with Contours and Ellipses ---
        axs[0, 0].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Image with Contours (Green) & Ellipses (Red)")
        axs[0, 0].axis('off')

        # --- Plot 2: Histogram of Vertical Slant ---
        axs[0, 1].hist(self.vertical_slants, bins=20, color='skyblue', alpha=0.8, edgecolor='black')
        axs[0, 1].axvline(
            x=metrics['vertical_slant'],
            color='r',
            linestyle='--',
            linewidth=2,
            label=f'Avg: {metrics["vertical_slant"]:.1f}°'
        )
        axs[0, 1].axvline(
            x=metrics['italic_threshold'],
            color='g',
            linestyle=':',
            linewidth=2,
            label=f'Italic Threshold: {metrics["italic_threshold"]}°'
        )
        axs[0, 1].set_title("Vertical Slant Distribution (Right tilt > 0)")
        axs[0, 1].set_xlabel("Vertical Slant (degrees from vertical)")
        axs[0, 1].set_ylabel("Frequency (Number of Components)")
        axs[0, 1].legend()
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.6)

        # --- Plot 3: Box Plot of Vertical Slant ---
        axs[1, 0].boxplot(self.vertical_slants, vert=False, showmeans=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue'),
                          medianprops=dict(color='red', linewidth=2),
                          meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='red'))
        axs[1, 0].axvline(
            x=metrics['italic_threshold'],
            color='g',
            linestyle=':',
            linewidth=2,
            label=f'Italic Threshold: {metrics["italic_threshold"]}°'
        )
        axs[1, 0].set_title("Vertical Slant Box Plot")
        axs[1, 0].set_xlabel("Vertical Slant (degrees from vertical)")
        axs[1, 0].set_yticks([])  # Hide y-axis ticks as it's just one box
        axs[1, 0].legend(loc='lower right')
        axs[1, 0].grid(axis='x', linestyle='--', alpha=0.6)

        # --- Plot 4: Metrics Summary ---
        axs[1, 1].axis('off')  # Hide axes
        axs[1, 1].set_title("Analysis Metrics", pad=20)
        metrics_text = (
            f"Avg. Vertical Slant: {metrics['vertical_slant']:.2f}°\n"
            f"Slant Std Dev: {metrics['slant_std']:.2f}°\n"
            f"Avg. Ellipse Angle: {metrics['avg_slant']:.2f}°\n"
            f"Components Found: {metrics['num_components']}\n"
            f"Italic Threshold: {metrics['italic_threshold']}°\n"
            f"Is Italic Detected: {'Yes' if metrics['is_italic'] else 'No'}"
        )
        # Add text centered in the subplot
        axs[1, 1].text(0.5, 0.5, metrics_text,
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        # plt.show() # Comment out for non-interactive use
        plt.close(fig)  # Close the figure to free memory

        return [plot_base64]  # Return as a list containing one plot

    def _encode_image_to_base64(self, image):
        """Encodes a given image to base64."""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze(self, debug=False, italic_threshold=8):
        """
        Performs the full slant angle analysis.

        Args:
            debug (bool): If True, generates and returns visualization plots.
            italic_threshold (float): The average vertical slant angle (in degrees)
                                      above which the text is considered italic.

        Returns:
            dict: A dictionary containing:
                  'metrics': A dict of calculated slant metrics.
                  'graphs': A list of base64 encoded PNG strings of the debug plots
                            (only if debug=True and data is available).
                  'preprocessed_image': base64 encoded preprocessed image.
        """
        self._preprocess_image()
        self.slant_angles, self.ellipses = self._calculate_slant_angles()  # Get ellipses too

        metrics = {}
        if self.slant_angles:
            # Vertical slant: Deviation from vertical (90 degrees).
            # Positive values indicate rightward tilt (typical italic).
            # Angle from fitEllipse is the rotation of the major axis from the x-axis.
            # We want the angle of the major axis from the *y-axis*.
            # If angle is 0-90, vertical slant = 90 - angle
            # If angle is 90-180, vertical slant = 90 - angle (will be negative, left tilt)
            # This seems correct assuming ellipse angle is counter-clockwise from x-axis
            self.vertical_slants = [90.0 - angle for angle in self.slant_angles]

            # Handle potential wrap-around if angles are near 0 or 180 - less common for text
            # For simplicity, the above calculation is often sufficient for typical text slants.

            avg_vertical_slant = float(np.mean(self.vertical_slants))  # Ensure float type
            slant_std = float(np.std(self.vertical_slants))  # Ensure float type
            avg_slant = float(np.mean(self.slant_angles))  # Ensure float type
            is_italic = avg_vertical_slant > italic_threshold

            metrics = {
                'avg_slant': avg_slant,  # Avg ellipse rotation (0-180)
                'vertical_slant': avg_vertical_slant,  # Avg angle from vertical (right tilt > 0)
                'slant_std': slant_std,  # Std dev of vertical slant
                'num_components': len(self.slant_angles),  # Number of text components analyzed
                'is_italic': is_italic,  # Boolean indicating italic detection
                'italic_threshold': float(italic_threshold)  # Threshold used
            }
        else:
            # Default metrics if no valid components found
            metrics = {
                'avg_slant': 0.0,
                'vertical_slant': 0.0,
                'slant_std': 0.0,
                'num_components': 0,
                'is_italic': False,
                'italic_threshold': float(italic_threshold)
            }

        result = {'metrics': metrics, 'graphs': []}

        if debug:
            result['graphs'] = self._generate_debug_plots(metrics)

        result['preprocessed_image'] = self._encode_image_to_base64(cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2BGR))

        return result