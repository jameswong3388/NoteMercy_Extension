import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class RightAngleAnalyzer:
    """
    Analyzes an image of a word to detect right-angle corners formed by
    sufficiently long line segments, characteristic of block letters.
    Uses fixed internal parameters for preprocessing and analysis.
    """

    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer by loading the original color image.
        """
        self.img_color = None
        if is_base64:
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            self.img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if self.img_color is None:
                raise ValueError("Could not decode base64 image as color.")
        else:
            self.img_color = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if self.img_color is None:
                raise ValueError(f"Could not read image as color at path: {image_input}")

        self.original_height, self.original_width = self.img_color.shape[:2]
        self._reset_analysis_data()

    def _reset_analysis_data(self):
        """
        Clears intermediate data and results from previous analysis runs.
        """
        self.binary_image = None
        self.contours = []
        self.approximated_polygons = []
        self.filtered_right_angle_corners = []  # [(x, y, angle_deg), ...]
        self.metrics = {}  # Store final results here

    def _preprocess_image(self):
        """
        Applies fixed preprocessing steps to get a binary image.
        """
        # --- Fixed Preprocessing Parameters ---
        BLUR_KSIZE = 17
        THRESHOLD_BLOCK_SIZE = 11
        THRESHOLD_C = 2
        MORPH_CLOSE_KSIZE = 3

        # 1. Grayscale
        gray_image = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        processed = gray_image.copy()

        # 2. Gaussian Blur
        if BLUR_KSIZE > 1:
            # Ensure ksize is odd
            ksize = BLUR_KSIZE if BLUR_KSIZE % 2 != 0 else BLUR_KSIZE + 1
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

        # 3. Adaptive Thresholding
        self.binary_image = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, THRESHOLD_BLOCK_SIZE, THRESHOLD_C
        )

        # 4. Morphological Closing
        if MORPH_CLOSE_KSIZE > 0:
            kernel = np.ones((MORPH_CLOSE_KSIZE, MORPH_CLOSE_KSIZE), np.uint8)
            self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel)

    def _calculate_angle(self, p1, p2, p3):
        """
        Calculates the angle (in degrees) at vertex p2 formed by p1-p2-p3.
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        if len_v1 == 0 or len_v2 == 0:
            return 180.0  # Treat collinear points as 180 degrees

        dot_product = np.dot(v1, v2)
        cos_angle = np.clip(dot_product / (len_v1 * len_v2), -1.0, 1.0)  # Clamp for stability
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def _analyze_contours(self):
        """
        Finds contours, approximates polygons, calculates vertex angles, AND
        filters for right-angle-like corners formed by sufficiently long segments,
        using fixed internal parameters.
        """
        # --- Fixed Analysis Parameters ---
        APPROX_EPSILON_FACTOR = 0.01    # Epsilon for polygon approximation (tune if needed)
        MIN_CONTOUR_LENGTH = 20         # Ignore contours smaller than this (pixels)
        MIN_SEGMENT_LENGTH_PIXELS = 10  # Min length for segments forming the corner (tune if needed)
        RIGHT_ANGLE_LOW_DEG = 80.0      # Lower bound for right-angle check
        RIGHT_ANGLE_HIGH_DEG = 100.0    # Upper bound for right-angle check

        # Find contours
        self.contours, _ = cv2.findContours(self.binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Reset results lists
        self.approximated_polygons = []
        self.filtered_right_angle_corners = []
        total_analyzed_contour_length = 0.0

        for cnt in self.contours:
            perimeter = cv2.arcLength(cnt, True)

            # Filter small contours (likely noise)
            if perimeter < MIN_CONTOUR_LENGTH:
                continue
            total_analyzed_contour_length += perimeter

            # Approximate contour with a polygon
            epsilon = APPROX_EPSILON_FACTOR * perimeter
            approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
            self.approximated_polygons.append(approx_poly)

            # Analyze angles if the polygon has at least 3 vertices
            if len(approx_poly) >= 3:
                num_vertices = len(approx_poly)
                for i in range(num_vertices):
                    # Get points forming the angle at vertex 'i'
                    p_prev = approx_poly[i - 1][0]  # Handles wrap-around
                    p_curr = approx_poly[i][0]
                    p_next = approx_poly[(i + 1) % num_vertices][0]  # Handles wrap-around

                    # Calculate lengths of the two segments forming the angle
                    len_prev_curr = np.linalg.norm(p_curr - p_prev)
                    len_curr_next = np.linalg.norm(p_next - p_curr)

                    # Apply segment length filter
                    if len_prev_curr >= MIN_SEGMENT_LENGTH_PIXELS and \
                            len_curr_next >= MIN_SEGMENT_LENGTH_PIXELS:

                        # Calculate angle only if segments are long enough
                        angle = self._calculate_angle(p_prev, p_curr, p_next)

                        # Check if angle is within the defined right-angle range
                        if RIGHT_ANGLE_LOW_DEG <= angle <= RIGHT_ANGLE_HIGH_DEG:
                            corner_data = (p_curr[0], p_curr[1], angle)
                            self.filtered_right_angle_corners.append(corner_data)

        # Store length for density calculation
        self.metrics['total_analyzed_contour_length'] = total_analyzed_contour_length

    def _calculate_metrics(self):
        """
        Calculates aggregate metrics based ONLY on filtered right-angle corners.
        """
        num_filtered_right = len(self.filtered_right_angle_corners)
        total_len = self.metrics.get('total_analyzed_contour_length', 0.0)

        self.metrics['right_angle_corner_count'] = num_filtered_right  # Simplified metric name

        # Calculate density (corners per 1000 pixels of contour length)
        if total_len > 0:
            self.metrics['right_angle_corner_density'] = (num_filtered_right / total_len) * 1000
        else:
            self.metrics['right_angle_corner_density'] = 0.0

    def _generate_visualization(self):
        """ Generates visualization showing polygons and filtered right-angle corners. """
        # --- Fixed Visualization Parameters ---
        FIGURE_SIZE = (15, 6)               # Overall figure size
        POLY_COLOR = (0, 255, 0)            # Green for polygon lines (BGR)
        RIGHT_CORNER_COLOR = (255, 0, 255)  # Magenta for corner markers (BGR)
        CORNER_MARKER_SIZE = 7              # Radius of corner markers
        LINE_THICKNESS = 1                  # Thickness for polygon lines
        LAYOUT_PADDING = 1.5                # Padding for subplot layout

        graphs = []  # List to hold base64 encoded graphs
        if self.img_color is None or self.binary_image is None:
            print("Warning: Cannot generate visualization, required data missing.")
            return graphs

        # --- Create Visualization Images ---
        # Background for polygons (start with binary, convert to BGR)
        vis_polygons = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis_polygons, self.approximated_polygons, isClosed=True, color=POLY_COLOR,
                      thickness=LINE_THICKNESS)

        # Background for corners (start with original color, convert to RGB for matplotlib)
        vis_corners = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        corner_count = len(self.filtered_right_angle_corners)
        # Draw filtered corners onto the color image copy
        for (x, y, angle) in self.filtered_right_angle_corners:
            cv2.circle(vis_corners, (x, y), CORNER_MARKER_SIZE, RIGHT_CORNER_COLOR, -1)  # Filled circle

        # --- Create Matplotlib Figure ---
        plt.figure("Right Angle Analysis", figsize=FIGURE_SIZE)

        # Plot 1: Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB))  # Ensure RGB for display
        plt.title("Original Image")
        plt.axis('off')

        # Plot 2: Polygons on Binary
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(vis_polygons, cv2.COLOR_BGR2RGB))  # Convert BGR polygon image to RGB
        plt.title("Approximated Polygons (Green)")
        plt.axis('off')

        # Plot 3: Detected Corners on Original
        plt.subplot(1, 3, 3)
        plt.imshow(vis_corners)  # Already RGB
        plt.title(f"Filtered ~90° Corners ({corner_count}, Magenta)")
        plt.axis('off')

        # Adjust layout and save to buffer
        plt.tight_layout(pad=LAYOUT_PADDING)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()  # Close the figure to free memory

        graphs.append(plot_base64)
        return graphs

    def analyze(self, debug=False):
        """
        Orchestrates the analysis using fixed internal parameters.
        """
        self._reset_analysis_data()  # Clear previous results
        self._preprocess_image()
        self._analyze_contours()  # Uses fixed internal parameters
        self._calculate_metrics()

        result = {'metrics': self.metrics.copy()}  # Return a copy of metrics

        # Add graphs only if debug is True and no error occurred
        if debug and 'error' not in self.metrics:
            try:
                graphs = self._generate_visualization()
                result['graphs'] = graphs
            except Exception as e:
                print(f"Error generating visualization: {e}")
                result['graphs'] = []  # Ensure graphs key exists but is empty
        else:
            result['graphs'] = []  # Ensure graphs key exists but is empty

        # Add preprocessed image
        _, preprocessed_image_buffer = cv2.imencode(".png", self.binary_image)
        preprocessed_image_base64 = base64.b64encode(preprocessed_image_buffer).decode('utf-8')
        result['preprocessed_image'] = preprocessed_image_base64

        return result