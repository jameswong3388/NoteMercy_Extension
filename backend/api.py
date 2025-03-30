from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

from helper import preprocess_image
from lib_py.block_lettering.angularity import AngularityAnalyzer
from lib_py.block_lettering.aspect_ratio import AspectRatioAnalyzer
from lib_py.block_lettering.loop_detection import LoopDetectionAnalyzer
from lib_py.calligraphic.continuous_part_coverage import ContinuousPartCoverageAnalyzer
from lib_py.calligraphic.right_angle_detection import RightAngleAnalyzer
from lib_py.calligraphic.stroke_width_variation import StrokeWidthAnalyzer
from lib_py.cursive.Curvature_Continuity import CursiveCurvatureAnalyzer
from lib_py.cursive.Enclosed_Loop_Ratio import EnclosedLoopAnalyzer
from lib_py.cursive.Stroke_Connectivity_Index import StrokeConnectivityAnalyzer
from lib_py.cursive.stroke_consistency import StrokeConsistencyAnalyzer
from lib_py.italic.Inter_Letter_Spacing_Uniformity import LetterSpacingAnalyzer
from lib_py.italic.Slant_Angle import SlantAngleAnalyzer
from lib_py.italic.Vertical_Stroke_Proportion import VerticalStrokeAnalyzer
from lib_py.print.discrete_letter import DiscreteLetterAnalyzer
from lib_py.print.letter_size_uniformity import LetterUniformityAnalyzer
from lib_py.print.vertical_alignment_consistency import VerticalAlignmentAnalyzer
from lib_py.shorthand.smooth_curves import StrokeSmoothnessAnalyzer
from lib_py.shorthand.stroke_continuity import StrokeContinuityAnalyzer
from lib_py.shorthand.symbol_density import SymbolDensityAnalyzer

app = FastAPI()

# Add CORS middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
    max_age=3600,
)


class ImageRequest(BaseModel):
    image: str  # base64 encoded image


@app.options("/api/v1/extract")
async def options_extract():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
    )


def convert_numpy_types(obj):
    """Convert numpy types to Python native types recursively."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


@app.post("/api/v1/extract")
async def analyze_image(request: ImageRequest):
    try:
        # Preprocess the image
        processed_image = preprocess_image(request.image)

        # - Feature Extraction - #
        # Block Lettering Analysis
        angularity_analyzer = AngularityAnalyzer(request.image, is_base64=True)
        angularity_results = angularity_analyzer.analyze(debug=True)

        aspect_ratio_analyzer = AspectRatioAnalyzer(request.image, is_base64=True)
        aspect_ratio_results = aspect_ratio_analyzer.analyze(debug=True)

        loop_detection_analyzer = LoopDetectionAnalyzer(request.image, is_base64=True)
        loop_detection_results = loop_detection_analyzer.analyze(debug=True)

        # Calligraphic Analysis
        coverage_analyzer = ContinuousPartCoverageAnalyzer(request.image, is_base64=True)
        coverage_results = coverage_analyzer.analyze(debug=True)

        right_angle_analyzer = RightAngleAnalyzer(request.image, is_base64=True)
        right_angle_results = right_angle_analyzer.analyze(debug=True)

        stroke_width_analyzer = StrokeWidthAnalyzer(request.image, is_base64=True)
        stroke_width_results = stroke_width_analyzer.analyze(debug=True)

        # Cursive Analysis
        curvature_analyzer = CursiveCurvatureAnalyzer(request.image, is_base64=True)
        curvature_results = curvature_analyzer.analyze(debug=True)

        loop_analyzer = EnclosedLoopAnalyzer(request.image, is_base64=True)
        loop_results = loop_analyzer.analyze(debug=True)

        connectivity_analyzer = StrokeConnectivityAnalyzer(request.image, is_base64=True)
        connectivity_results = connectivity_analyzer.analyze(debug=True)

        consistency_analyzer = StrokeConsistencyAnalyzer(request.image, is_base64=True)
        consistency_results = consistency_analyzer.analyze(debug=True)

        # Italic Analysis
        spacing_analyzer = LetterSpacingAnalyzer(request.image, is_base64=True)
        spacing_results = spacing_analyzer.analyze(debug=True)

        slant_angle_analyzer = SlantAngleAnalyzer(request.image, is_base64=True)
        slant_angle_results = slant_angle_analyzer.analyze(debug=True)

        vertical_stroke_analyzer = VerticalStrokeAnalyzer(request.image, is_base64=True)
        vertical_stroke_results = vertical_stroke_analyzer.analyze(debug=True)

        # Print Analysis
        discrete_letter_analyzer = DiscreteLetterAnalyzer(request.image, is_base64=True)
        discrete_letter_results = discrete_letter_analyzer.analyze(debug=True)

        letter_size_analyzer = LetterUniformityAnalyzer(request.image, is_base64=True)
        letter_size_results = letter_size_analyzer.analyze(debug=True)

        vertical_alignment_analyzer = VerticalAlignmentAnalyzer(request.image, is_base64=True)
        vertical_alignment_results = vertical_alignment_analyzer.analyze(debug=True)

        # Shorthand Analysis
        continuity_analyzer = StrokeContinuityAnalyzer(request.image, is_base64=True)
        continuity_results = continuity_analyzer.analyze(debug=True)

        smooth_curves_analyzer = StrokeSmoothnessAnalyzer(request.image, is_base64=True)
        smooth_curves_results = smooth_curves_analyzer.analyze(debug=True)

        symbol_density_analyzer = SymbolDensityAnalyzer(request.image, is_base64=True)
        symbol_density_results = symbol_density_analyzer.analyze(debug=True)

        # =====================================================
        # === SCORE CALCULATION FOR BLOCK LETTERING STYLE ===
        # =====================================================
        # Retrieve metrics from the analyzers
        angularity_metrics = angularity_results.get('metrics', {})
        aspect_ratio_metrics = aspect_ratio_results.get('metrics', {})
        loop_detection_metrics = loop_detection_results.get('metrics', {})

        # Angularity
        median_angle = angularity_metrics.get('median_turning_angle')
        shape_count = angularity_metrics.get('shape_count', 0)

        angularity_score = 0.0  # Default score
        if shape_count > 0 and median_angle is not None and median_angle > 0:
            if 80.0 <= median_angle <= 100.0:
                angularity_score = 1.0
            elif median_angle >= 135.0 or median_angle <= 45.0:
                angularity_score = 0.0
            else:
                # Linear interpolation between the thresholds: score decreases as angle increases
                angularity_score = (135.0 - median_angle) / 90.0
                angularity_score = max(0.0, min(1.0, angularity_score))

        # Aspect ratio score
        std_dev_ar = aspect_ratio_metrics.get('std_dev_aspect_ratio')
        num_candidates = aspect_ratio_metrics.get('num_letter_candidates', 0)
        aspect_ratio_consistency_score = 0.0

        if num_candidates >= 2 and std_dev_ar is not None:
            if std_dev_ar <= 0.15:
                aspect_ratio_consistency_score = 1.0
            elif std_dev_ar >= 1.0:
                aspect_ratio_consistency_score = 0.0
            else:
                # Linear interpolation: score decreases as std dev increases
                aspect_ratio_consistency_score = (1.0 - std_dev_ar) / (
                            1.0 - 0.15)
                aspect_ratio_consistency_score = max(0.0, min(1.0, aspect_ratio_consistency_score))

        # Loop percentage score
        percentage_loops = loop_detection_metrics.get('percentage_shapes_with_loops', 0.0)
        # Convert percentage to a score between 0 and 1
        loop_score = 1 - (percentage_loops / 100.0)
        loop_score = max(0.0, max(0.0, loop_score))

        # Combined block lettering score (equal weights)
        block_lettering_score = (angularity_score + aspect_ratio_consistency_score + loop_score) / 3

        # =====================================================
        # === SCORE CALCULATION FOR CALLIGRAPHIC HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics from the analyzers
        coverage_metrics = coverage_results.get('metrics', {})
        right_angle_metrics = right_angle_results.get('metrics', {})
        stroke_width_metrics = stroke_width_results.get('metrics', {})

        # Continuous part coverage score
        coverage_ratio = coverage_metrics.get('continuous_part_coverage_ratio', 0.0)
        coverage_score = max(0.0, min(1.0, coverage_ratio))

        # Right angle density score
        density = right_angle_metrics.get('right_angle_corner_density', 0.0)
        right_angle_score = 0.0

        MAX_SCORE_DENSITY = 10.0 # Threshold: Densities >= this value will get score 0

        if MAX_SCORE_DENSITY > 0:
            # Simple linear scaling down from 1 (at density 0) to 0 (at density MAX_SCORE_DENSITY)
            right_angle_score = 1.0 - (density / MAX_SCORE_DENSITY)
            right_angle_score = max(0.0, min(1.0, right_angle_score))  # Clamp score to [0, 1]
        else:
            right_angle_score = 1.0 if density == 0 else 0.0

        # Stroke width variation score
        variation_coefficient = stroke_width_metrics.get('variation_coefficient', 0.0)
        width_score = 0.0  # Initialize the score

        # Define a threshold for CV where the score reaches 1.0 (TUNABLE PARAMETER)
        # A CV of 0.5 might indicate significant variation. Adjust based on testing.
        MAX_SCORE_CV_THRESHOLD = 0.6

        if MAX_SCORE_CV_THRESHOLD > 0:
            # Linearly scale the score: 0 at CV=0, 1 at CV=THRESHOLD
            width_score = variation_coefficient / MAX_SCORE_CV_THRESHOLD
            # Clamp the score to be between 0.0 and 1.0
            width_score = max(0.0, min(1.0, width_score))
        else:
            # Handle edge case of zero threshold
            width_score = 1.0 if variation_coefficient > 0 else 0.0

        # Combined calligraphic score (equal weights)
        calligraphic_score = (coverage_score + right_angle_score + width_score) / 3

        # =====================================================
        # === SCORE CALCULATION FOR CURSIVE HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics from the analyzers
        curvature_metrics = curvature_results.get('metrics', {})
        loop_metrics = loop_results.get('metrics', {})
        connectivity_metrics = connectivity_results.get('metrics', {})
        consistency_metrics = consistency_results.get('metrics', {})

        # Curvature Score Calculation
        # Use average normalized segment length from CursiveCurvatureAnalyzer.
        # Hypothesis: Smoother cursive curves are approximated by longer segments.
        # We normalize this length against a threshold to get a score from 0 to 1.
        avg_norm_seg_len = curvature_metrics.get('avg_normalized_segment_length', 0.0)

        # Define a threshold for average segment length (as fraction of image height)
        # that corresponds to a maximum score (e.g., 1.0). This is a tunable parameter.
        # Lengths above this will also get a score of 1.0.
        # Example: If avg segment length is 5% of image height, score is 1.0.
        AVG_LENGTH_THRESHOLD_FOR_MAX_SCORE = 0.05  # Tunable: adjust based on testing

        if AVG_LENGTH_THRESHOLD_FOR_MAX_SCORE > 0:
            # Linearly scale the score from 0 up to the threshold.
            curvature_score = min(1.0, max(0.0, avg_norm_seg_len / AVG_LENGTH_THRESHOLD_FOR_MAX_SCORE))
        else:
            # Avoid division by zero; if threshold is 0, score is 1 only if length is > 0.
            curvature_score = 1.0 if avg_norm_seg_len > 0 else 0.0

        # Lower 'average_components_per_word' means MORE connected (more cursive).
        # We need to map this to a score where 1 is highly connected and 0 is disconnected.

        # Define thresholds based on typical observations or the classify_connectivity function:
        # - A word being a single component (avg_comps=1) is highly cursive. Score = 1.
        # - Words averaging many components (e.g., >= 8) are very print-like. Score = 0.
        MIN_COMPONENTS_FOR_MAX_SCORE = 1.0  # Below this, score is 1
        MAX_COMPONENTS_FOR_MIN_SCORE = 8.0  # At or above this, score is 0

        avg_comps_per_word = connectivity_metrics.get('average_components_per_word', None)
        word_count = connectivity_metrics.get('word_count', 0)

        connectivity_score = 0.0  # Default score if no words or components detected

        if word_count > 0 and avg_comps_per_word is not None:
            if avg_comps_per_word <= MIN_COMPONENTS_FOR_MAX_SCORE:
                connectivity_score = 1.0
            elif avg_comps_per_word >= MAX_COMPONENTS_FOR_MIN_SCORE:
                connectivity_score = 0.0
            else:
                # Linear interpolation between the thresholds
                # Score decreases as avg_comps_per_word increases
                connectivity_score = (MAX_COMPONENTS_FOR_MIN_SCORE - avg_comps_per_word) / (MAX_COMPONENTS_FOR_MIN_SCORE - MIN_COMPONENTS_FOR_MAX_SCORE)
                # Ensure score is clamped just in case (shouldn't be needed with checks above)
                connectivity_score = min(1.0, max(0.0, connectivity_score))

        # Enclosed loop ratio: higher ratio indicates more cursive features
        loop_ratio = loop_metrics.get('enclosed_loop_ratio', 0)
        loop_score = min(1, max(0, loop_ratio))

        # Consistency Score
        consistency_score = consistency_metrics.get('stroke_consistency_index', 0)

        # Combined cursive score (equal weights)
        cursive_score = (connectivity_score + loop_score + curvature_score + consistency_score) / 4

        # =====================================================
        # === SCORE CALCULATION FOR ITALIC HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics from the analyzers
        spacing_metrics = spacing_results.get('metrics', {})
        slant_metrics = slant_angle_results.get('metrics', {})
        vertical_stroke_metrics = vertical_stroke_results.get('metrics', {})

        # Letter spacing: uniform spacing contributes to italic appearance
        spacing_is_uniform = spacing_metrics.get('is_uniform')
        if spacing_is_uniform is None:
            # Not enough gaps to determine uniformity; assign a neutral score.
            spacing_score = 0.5
        else:
            spacing_score = 1.0 if spacing_is_uniform else 0.0

        # Slant angle: higher angle (within range) indicates italic writing
        vertical_slant = abs(slant_metrics.get('vertical_slant', 0))
        italic_threshold = slant_metrics.get('italic_threshold', 8)
        slant_std = slant_metrics.get('slant_std', 0)

        slant_score = min(1.0, vertical_slant / (italic_threshold + 5))
        if slant_std > 10:
            slant_score *= 0.85  # penalize unstable slants

        # --- Vertical Stroke Proportion Score ---
        # Using metrics from the new VerticalStrokeAnalyzer: median_height, max_height, ascender_ratio
        # A higher ascender_ratio (max_height / median_height) indicates more pronounced difference
        # between x-height and ascender/capital height, common in italic, cursive, calligraphic styles.
        # We assume that for italic, a significant ratio (e.g., > 1.5) is more characteristic than a ratio near 1 (like print/block).
        # Let's normalize the score based on the ascender_ratio.
        ascender_ratio = vertical_stroke_metrics.get('ascender_ratio',
                                                     1.0)  # Default to 1 (uniform) if not found or median_height is 0

        # Define thresholds for scoring:
        baseline_ratio = 1.2  # Ratios below this (close to uniform height) get low scores.
        target_ratio_for_max_score = 3.0  # Ratios at or above this get a score of 1. Ratios between baseline and target are scaled linearly.

        if ascender_ratio <= baseline_ratio:
            vertical_score = 0.0  # Ratio indicates uniform height, less likely italic
        elif ascender_ratio >= target_ratio_for_max_score:
            vertical_score = 1.0  # Ratio is high, consistent with distinct vertical zones
        else:
            # Linear scaling between baseline and target
            vertical_score = (ascender_ratio - baseline_ratio) / (target_ratio_for_max_score - baseline_ratio)

        # Ensure score is clamped between 0 and 1 (though logic above should handle it)
        vertical_score = min(1.0, max(0.0, vertical_score))

        # Combined italic score (equal weights)
        italic_score = (vertical_score + slant_score + spacing_score) / 3

        # =====================================================
        # === SCORE CALCULATION FOR PRINT HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics from the analyzers
        vertical_alignment_metrics = vertical_alignment_results.get('metrics', {})
        letter_size_metrics = letter_size_results.get('metrics', {})
        discrete_letter_metrics = discrete_letter_results.get('metrics', {})

        # Vertical alignment: higher alignment indicates print writing
        overall_align_score = vertical_alignment_metrics.get('overall_alignment_score', 0.0)
        height_consistency = vertical_alignment_metrics.get('height_consistency', 1.0)  # Default to max inconsistency
        consistency_score = max(0.0, 1.0 - height_consistency)  # Higher score = better consistency

        # Combine alignment and consistency (e.g., average them)
        combined_alignment_score = (overall_align_score + consistency_score) / 2.0

        # Letter size uniformity: higher uniformity indicates print
        height_uniformity = letter_size_metrics.get('height_uniformity', 0)
        width_uniformity = letter_size_metrics.get('width_uniformity', 0)
        aspect_ratio_uniformity = letter_size_metrics.get('aspect_ratio_uniformity', 0)
        size_uniformity = (height_uniformity + width_uniformity + aspect_ratio_uniformity) / 3
        size_score = min(1, max(0, size_uniformity))

        # Discrete letters: higher discreteness indicates print over cursive
        num_components = discrete_letter_metrics.get('num_components', 0)
        total_components = discrete_letter_metrics.get('total_components', 0)
        discrete_index = num_components / total_components if total_components > 0 else 0
        discrete_score = min(1, max(0, discrete_index))

        # Combined print score (equal weights)
        print_score = (combined_alignment_score + size_score + discrete_score) / 3

        # =====================================================
        # === SCORE CALCULATION FOR SHORTHAND HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics from the analyzers
        smooth_curves_metrics = smooth_curves_results.get('metrics', {})
        continuity_metrics = continuity_results.get('metrics', {})
        symbol_density_metrics = symbol_density_results.get('metrics', {})

        # Smooth curves: higher smoothness indicates shorthand writing
        avg_angle_change = smooth_curves_metrics.get('avg_abs_angle_change', 1.0)  # Default assumes very unsmooth

        # Define a threshold for max average angle change (in radians) considered "smooth enough".
        # e.g., 0.5 radians is about 28.6 degrees average change between spline segments.
        # A lower threshold makes the smoothness requirement stricter. Needs tuning.
        SMOOTHNESS_THRESHOLD_RAD = 0.3

        # Calculate score: linearly decrease from 1 (at 0 change) to 0 (at threshold).
        # score = 1 - (value / threshold)
        curve_score = max(0.0, 1.0 - (avg_angle_change / SMOOTHNESS_THRESHOLD_RAD))

        # Stroke continuity: higher continuity indicates shorthand
        # Using metrics from the updated StrokeContinuityAnalyzer
        num_endpoints = continuity_metrics.get('num_endpoints', 0)
        num_components = continuity_metrics.get('num_components', 0)

        if num_components > 0:
            # Calculate ratio of endpoints per component.
            # Lower values (e.g., approaching 0 for loops, 2 for simple lines) suggest more continuity.
            # Higher values suggest more breaks or complex/fragmented components.
            endpoint_ratio = num_endpoints / num_components

            # Define a threshold for the ratio. Ratios above this are considered highly discontinuous.
            # A higher threshold is more lenient. Needs tuning based on expected shorthand characteristics.
            # Example: If a ratio of 6 (e.g., 3 separate short lines) is considered discontinuous.
            CONTINUITY_ENDPOINT_THRESHOLD = 6.0

            # Calculate score: Higher score for lower ratios (more continuous).
            # Linearly decreases from 1 (at ratio 0) to 0 (at or above the threshold).
            continuity_score = max(0.0, 1.0 - (endpoint_ratio / CONTINUITY_ENDPOINT_THRESHOLD))
        else:
            # No components found (e.g., blank image or processing error), implies zero continuity.
            continuity_score = 0.0

        # Symbol density: higher density indicates shorthand
        density_index = symbol_density_metrics.get('density_index', 0)
        density_score = min(1, max(0, density_index))

        # Combined shorthand score (equal weights)
        shorthand_score = (continuity_score + curve_score + density_score) / 3


        # Convert all results to Python native types
        response = {
            "processed_image": processed_image,
            # block lettering
            "angularity": convert_numpy_types(angularity_results),
            "aspect_ratio": convert_numpy_types(aspect_ratio_results),
            "loop_detection": convert_numpy_types(loop_detection_results),

            # calligraphic
            "continuous_part_coverage": convert_numpy_types(coverage_results),
            "right_angle_corner_detection": convert_numpy_types(right_angle_results),
            "stroke_width_variation": convert_numpy_types(stroke_width_results),

            # cursive
            "stroke_connectivity": convert_numpy_types(connectivity_results),
            "enclosed_loop_ratio": convert_numpy_types(loop_results),
            "curvature_continuity": convert_numpy_types(curvature_results),
            "stroke_consistency": convert_numpy_types(consistency_results),

            # italic
            "vertical_stroke_proportion": convert_numpy_types(vertical_stroke_results),
            "slant_angle": convert_numpy_types(slant_angle_results),
            "inter_letter_spacing": convert_numpy_types(spacing_results),

            # print
            "vertical_alignment": convert_numpy_types(vertical_alignment_results),
            "letter_size_uniformity": convert_numpy_types(letter_size_results),
            "discrete_letter": convert_numpy_types(discrete_letter_results),

            # shorthand
            "stroke_continuity": convert_numpy_types(continuity_results),
            "smooth_curves": convert_numpy_types(smooth_curves_results),
            "symbol_density": convert_numpy_types(symbol_density_results),


            "handwriting": {
                "block_lettering": {
                    "score": float(block_lettering_score),
                },
                "cursive": {
                    "score": float(cursive_score),
                },
                "calligraphic": {
                    "score": float(calligraphic_score),
                },
                "italic": {
                    "score": float(italic_score),
                },
                "shorthand": {
                    "score": float(shorthand_score),
                },
                "print": {
                    "score": float(print_score),
                },
            }
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
