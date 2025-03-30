import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import math


class AngularityAnalyzer:
    def __init__(self, image_input, is_base64=True):
        """
        Initializes the analyzer by loading the image.

        Parameters:
            image_input (str): Base64 encoded image string or image file path.
            is_base64 (bool): True if image_input is base64, False if it's a path.
        """
        self.img = None
        if is_base64:
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Load as color
            if self.img is None:
                raise ValueError("Could not decode base64 image.")
        else:
            self.img = cv2.imread(image_input, cv2.IMREAD_COLOR)  # Load as color
            if self.img is None:
                raise ValueError(f"Could not read image at path: {image_input}")

        self.original_height, self.original_width = self.img.shape[:2]

        # --- Attributes to store intermediate and final results ---
        self.gray_image = None
        self.binary_image = None
        self.contours = []
        self.simplified_polygons = []
        # Store aggregated raw data for statistics calculation
        self.all_turning_angles = []
        self.all_circularities = []
        self.all_vertex_densities = []
        self.all_areas = []

    def _reset_analysis_data(self):
        """Clears data from previous analysis runs."""
        self.contours = []
        self.simplified_polygons = []
        self.all_turning_angles = []
        self.all_circularities = []
        self.all_vertex_densities = []
        self.all_areas = []

    def preprocess_image(self, blur_ksize=5):
        """
        Applies preprocessing: Grayscale, Blur, Thresholding.
        Stores results in self.gray_image and self.binary_image.

        Parameters:
            blur_ksize (int): Kernel size for Gaussian Blur (odd number > 1). 0 or 1 disables blur.
        """
        if self.img is None:
            # This should ideally not happen if __init__ succeeded
            raise RuntimeError("Cannot preprocess: Original image not loaded.")

        # 1. Convert to Grayscale
        if len(self.img.shape) == 3:
            self.gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:  # Assume already grayscale
            self.gray_image = self.img.copy()

        processed = self.gray_image.copy()

        # 2. Noise Reduction (Optional)
        if blur_ksize > 1:
            ksize = blur_ksize if blur_ksize % 2 != 0 else blur_ksize + 1
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

        # 3. Binarization (Otsu's method)
        _, self.binary_image = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Optional Morphological ops could modify self.binary_image here

    def _find_and_analyze_contours(self, approx_epsilon_factor=0.01, contour_min_area=50):
        """
        Finds contours in self.binary_image, analyzes their geometry,
        and stores aggregated data (angles, circularities, etc.) and simplified polygons.
        """
        if self.binary_image is None:
            raise RuntimeError("Cannot find contours: Preprocessing not done or failed.")

        # Find contours
        self.contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Reset lists before populating
        self.simplified_polygons = []
        self.all_turning_angles = []
        self.all_circularities = []
        self.all_vertex_densities = []
        self.all_areas = []

        # Analyze each contour
        for cnt in self.contours:
            analysis_result = self._analyze_contour_geometry(
                cnt,
                epsilon_factor=approx_epsilon_factor,
                min_area=contour_min_area
            )
            if analysis_result:
                simplified_poly, angles, circularity, vertex_density, area = analysis_result
                # Aggregate results from valid contours
                self.simplified_polygons.append(simplified_poly)
                self.all_turning_angles.extend(angles)
                self.all_circularities.append(circularity)
                self.all_vertex_densities.append(vertex_density)
                self.all_areas.append(area)

    # --- Contour Geometry Helper Methods (largely unchanged) ---

    def _calculate_contour_metrics(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        return area, perimeter

    def _calculate_circularity(self, area, perimeter):
        if perimeter == 0: return 0.0
        return (4 * math.pi * area) / (perimeter ** 2)

    def _calculate_turning_angles(self, polygon):
        angles = []
        num_points = polygon.shape[0]
        if num_points < 3: return []
        for i in range(num_points):
            p1 = polygon[(i - 1 + num_points) % num_points]
            p2 = polygon[i]
            p3 = polygon[(i + 1) % num_points]
            v1, v2 = p1 - p2, p3 - p2
            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0: continue
            dot_product = np.dot(v1, v2)
            cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)
        return angles

    def _analyze_contour_geometry(self, contour, epsilon_factor=0.01, min_area=50):
        area, perimeter = self._calculate_contour_metrics(contour)
        if area < min_area or perimeter == 0: return None
        circularity = self._calculate_circularity(area, perimeter)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified_poly = approx.reshape(-1, 2)
        if simplified_poly.shape[0] < 3: return None
        turning_angles = self._calculate_turning_angles(simplified_poly)
        vertex_count = simplified_poly.shape[0]
        vertex_density = vertex_count / perimeter
        return simplified_poly, turning_angles, circularity, vertex_density, area

    def _calculate_statistics(self):
        """
        Calculates summary statistics based on the aggregated contour data
        and stores them in self.metrics.
        """
        total_vertices = len(self.all_turning_angles)
        shape_count = len(self.simplified_polygons)  # Use count of successfully analyzed polygons

        metrics = {
            'avg_turning_angle': np.mean(self.all_turning_angles) if total_vertices > 0 else 0,
            'std_dev_turning_angle': np.std(self.all_turning_angles) if total_vertices > 0 else 0,
            'median_turning_angle': np.median(self.all_turning_angles) if total_vertices > 0 else 0,

            'avg_circularity': np.mean(self.all_circularities) if shape_count > 0 else 0,
            'std_dev_circularity': np.std(self.all_circularities) if shape_count > 0 else 0,
            'median_circularity': np.median(self.all_circularities) if shape_count > 0 else 0,

            'avg_vertex_density': np.mean(self.all_vertex_densities) if shape_count > 0 else 0,
            'avg_area': np.mean(self.all_areas) if shape_count > 0 else 0,
            'total_area': np.sum(self.all_areas),

            'total_vertices': total_vertices,
            'shape_count': shape_count,  # Count of analyzed shapes meeting criteria
            'avg_vertices_per_shape': total_vertices / shape_count if shape_count > 0 else 0
        }
        return metrics

    def _generate_visualization(self, pp_blur_ksize, approx_epsilon_factor, metrics):
        """
        Generates the visualization plot using stored data (images, polygons, metrics)
        and appends the base64 encoded graph to self.graphs.
        """
        graphs = []

        # Prepare images for plotting
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        plt.figure("Handwriting Style Analysis", figsize=(12, 10))

        # Plot 1: Original Image
        plt.subplot(2, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis('off')

        # Plot 2: Preprocessed Binary Image
        plt.subplot(2, 2, 2);
        plt.imshow(self.binary_image, cmap='gray')
        plt.title(f"Preprocessed (Blur K={pp_blur_ksize}, Otsu)")
        plt.axis('off')

        # Plot 3: Simplified Polygons on Original
        plt.subplot(2, 2, 3)
        plt.imshow(img_rgb)
        for poly in self.simplified_polygons:
            poly_closed = np.vstack([poly, poly[0]])
            plt.plot(poly_closed[:, 0], poly_closed[:, 1], 'r-', linewidth=1)
        plt.title(f"Simplified Polygons (Eps Factor: {approx_epsilon_factor})")
        plt.axis('off')

        # Plot 4: Turning Angle Histogram
        plt.subplot(2, 2, 4)
        if self.all_turning_angles:
            plt.hist(self.all_turning_angles, bins=30, range=(0, 180))
            plt.axvline(metrics['avg_turning_angle'], color='r', ls='--', lw=2,
                        label=f"Mean: {metrics['avg_turning_angle']:.1f}°")
            plt.axvline(metrics['median_turning_angle'], color='g', ls=':', lw=2,
                        label=f"Median: {metrics['median_turning_angle']:.1f}°")
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No valid angles found', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f"Turning Angle Dist. (Vertices: {metrics['total_vertices']})")
        plt.xlabel("Angle (°)")
        plt.ylabel("Frequency")
        plt.xlim(0, 180)

        plt.tight_layout()

        # Save plot to buffer and encode
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()  # Close the figure

        graphs.append(plot_base64)
        return graphs

    def analyze(self, debug=False, approx_epsilon_factor=0.01, pp_blur_ksize=5, contour_min_area=50):
        """
        Orchestrates the analysis process: preprocess, find/analyze contours,
        calculate statistics, and optionally generate visualization.

        Parameters:
            debug (bool): Generate visualization plots if True.
            approx_epsilon_factor (float): Controls polygon simplification detail.
            pp_blur_ksize (int): Kernel size for preprocessing blur.
            contour_min_area (float): Minimum area for contours to be analyzed.

        Returns:
            dict: Contains 'metrics' dictionary and 'graphs' list (if debug=True).
        """
        # --- 0. Reset data from previous runs ---
        self._reset_analysis_data()

        # --- 1. Preprocess ---
        self.preprocess_image(blur_ksize=pp_blur_ksize)

        # --- 2. Find and Analyze Contours ---
        self._find_and_analyze_contours(
            approx_epsilon_factor=approx_epsilon_factor,
            contour_min_area=contour_min_area
        )

        # --- 3. Calculate Statistics ---
        metrics = self._calculate_statistics()

        result = {
            'metrics': metrics,
            'graphs': []
        }

        # --- 4. Generate Visualization (Optional) ---
        if debug:
            result['graphs'] =  self._generate_visualization(
                pp_blur_ksize=pp_blur_ksize,
                approx_epsilon_factor=approx_epsilon_factor,
                metrics=metrics
            )

        # --- 5. Return Results ---
        return result

# === Example usage (remains the same) ===
if __name__ == "__main__":
    # --- Configuration ---
    image_path = r"C:\Users\Samson\Desktop\Coding\IPPR\NoteMercy_Extension\backend\atest\print2.png"  # Replace
    analyzer = AngularityAnalyzer(image_path, is_base64=False)
    results = analyzer.analyze(debug=True)

    print("\n===== Angularity Analysis Results =====")
    metrics = results['metrics']
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Display the image directly without saving
    if results['graphs']:
        from PIL import Image
        import io

        print("\nDisplaying visualization...")
        img_data = base64.b64decode(results['graphs'][0])
        img = Image.open(io.BytesIO(img_data))
        img.show()  # This will open the image in your default image viewer

