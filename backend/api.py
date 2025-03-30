import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
    allow_origins=["http://localhost:3000"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow POST and OPTIONS for preflight
    allow_headers=["Content-Type", "Authorization"],  # Allow necessary headers
    expose_headers=["Content-Type"],  # Expose headers if needed by frontend
    max_age=3600,  # Cache preflight response for 1 hour
)


class ImageRequest(BaseModel):
    image: str  # base64 encoded image


# Define OPTIONS route handler for CORS preflight requests
@app.options("/api/v1/extract")
async def options_extract():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",  # Match your frontend origin
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",  # Explicitly allow Content-Type
            "Access-Control-Max-Age": "3600",  # Cache preflight response
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
        processed_image_base64 = preprocess_image(request.image)

        # - Feature Extraction - #
        # Instantiate all analyzers with the original base64 image data
        angularity_analyzer = AngularityAnalyzer(request.image, is_base64=True)
        aspect_ratio_analyzer = AspectRatioAnalyzer(request.image, is_base64=True)
        loop_detection_analyzer = LoopDetectionAnalyzer(request.image, is_base64=True)
        coverage_analyzer = ContinuousPartCoverageAnalyzer(request.image, is_base64=True)
        right_angle_analyzer = RightAngleAnalyzer(request.image, is_base64=True)
        stroke_width_analyzer = StrokeWidthAnalyzer(request.image, is_base64=True)
        curvature_analyzer = CursiveCurvatureAnalyzer(request.image, is_base64=True)
        loop_analyzer = EnclosedLoopAnalyzer(request.image, is_base64=True)
        connectivity_analyzer = StrokeConnectivityAnalyzer(request.image, is_base64=True)
        consistency_analyzer = StrokeConsistencyAnalyzer(request.image, is_base64=True)
        spacing_analyzer = LetterSpacingAnalyzer(request.image, is_base64=True)
        slant_angle_analyzer = SlantAngleAnalyzer(request.image, is_base64=True)
        vertical_stroke_analyzer = VerticalStrokeAnalyzer(request.image, is_base64=True)
        discrete_letter_analyzer = DiscreteLetterAnalyzer(request.image, is_base64=True)
        letter_size_analyzer = LetterUniformityAnalyzer(request.image, is_base64=True)
        vertical_alignment_analyzer = VerticalAlignmentAnalyzer(request.image, is_base64=True)
        shorthand_continuity_analyzer = StrokeContinuityAnalyzer(request.image, is_base64=True)  # Renamed variable
        smooth_curves_analyzer = StrokeSmoothnessAnalyzer(request.image, is_base64=True)
        symbol_density_analyzer = SymbolDensityAnalyzer(request.image, is_base64=True)

        # Run analysis (consider parallelizing if performance is critical)
        angularity_results = angularity_analyzer.analyze(debug=True)
        aspect_ratio_results = aspect_ratio_analyzer.analyze(debug=True)
        loop_detection_results = loop_detection_analyzer.analyze(debug=True)
        coverage_results = coverage_analyzer.analyze(debug=True)
        right_angle_results = right_angle_analyzer.analyze(debug=True)
        stroke_width_results = stroke_width_analyzer.analyze(debug=True)
        curvature_results = curvature_analyzer.analyze(debug=True)
        loop_results = loop_analyzer.analyze(debug=True)
        connectivity_results = connectivity_analyzer.analyze(debug=True)
        consistency_results = consistency_analyzer.analyze(debug=True)
        spacing_results = spacing_analyzer.analyze(debug=True)
        slant_angle_results = slant_angle_analyzer.analyze(debug=True)
        vertical_stroke_results = vertical_stroke_analyzer.analyze(debug=True)
        discrete_letter_results = discrete_letter_analyzer.analyze(debug=True)
        letter_size_results = letter_size_analyzer.analyze(debug=True)
        vertical_alignment_results = vertical_alignment_analyzer.analyze(debug=True)
        shorthand_continuity_results = shorthand_continuity_analyzer.analyze(debug=True)  # Renamed variable
        smooth_curves_results = smooth_curves_analyzer.analyze(debug=True)
        symbol_density_results = symbol_density_analyzer.analyze(debug=True)


        # =====================================================
        # === SCORE CALCULATION FOR BLOCK LETTERING STYLE ===
        # =====================================================
        # Retrieve metrics
        block_angularity_metrics = angularity_results.get('metrics', {})
        block_aspect_ratio_metrics = aspect_ratio_results.get('metrics', {})
        block_loop_metrics = loop_detection_results.get('metrics', {})

        # --- Angularity Score ---
        block_median_angle = block_angularity_metrics.get('median_turning_angle')
        block_shape_count = block_angularity_metrics.get('shape_count', 0)
        block_angularity_feature_score = 0.0  # Default score
        if block_shape_count > 0 and block_median_angle is not None and block_median_angle > 0:
            if 80.0 <= block_median_angle <= 100.0:  # Ideal range for blocky letters
                block_angularity_feature_score = 1.0
            elif block_median_angle >= 135.0 or block_median_angle <= 45.0:  # Very curved or very sharp (less blocky)
                block_angularity_feature_score = 0.0
            else:
                # Linear interpolation: score decreases as angle moves away from 90 degrees towards 135 or 45
                # Simplified: Score decreases linearly as angle moves towards 135 (away from 90)
                # and also decreases linearly as angle moves towards 45 (away from 90)
                # Let's use a simpler approach: higher score for angles closer to 90.
                # Max deviation is 45 degrees (90 +/- 45). Score = 1 - (abs(angle - 90) / 45)
                deviation_from_90 = abs(block_median_angle - 90.0)
                block_angularity_feature_score = max(0.0, 1.0 - (deviation_from_90 / 45.0))

        # --- Aspect Ratio Consistency Score ---
        block_aspect_ratio_std_dev = block_aspect_ratio_metrics.get('std_dev_aspect_ratio')
        block_num_letter_candidates = block_aspect_ratio_metrics.get('num_letter_candidates', 0)
        block_aspect_ratio_consistency_score = 0.0  # Default
        # Thresholds for standard deviation (lower is more consistent/block-like)
        BLOCK_AR_STD_DEV_MAX_SCORE_THRESHOLD = 0.15  # Below this -> score 1.0
        BLOCK_AR_STD_DEV_MIN_SCORE_THRESHOLD = 1.0  # Above this -> score 0.0
        if block_num_letter_candidates >= 2 and block_aspect_ratio_std_dev is not None:
            if block_aspect_ratio_std_dev <= BLOCK_AR_STD_DEV_MAX_SCORE_THRESHOLD:
                block_aspect_ratio_consistency_score = 1.0
            elif block_aspect_ratio_std_dev >= BLOCK_AR_STD_DEV_MIN_SCORE_THRESHOLD:
                block_aspect_ratio_consistency_score = 0.0
            else:
                # Linear interpolation between thresholds
                block_aspect_ratio_consistency_score = (
                                                               BLOCK_AR_STD_DEV_MIN_SCORE_THRESHOLD - block_aspect_ratio_std_dev) / \
                                                       (
                                                               BLOCK_AR_STD_DEV_MIN_SCORE_THRESHOLD - BLOCK_AR_STD_DEV_MAX_SCORE_THRESHOLD)
                block_aspect_ratio_consistency_score = max(0.0, min(1.0, block_aspect_ratio_consistency_score))

        # --- Loop Presence Score (Inverse) ---
        # Block letters tend to have fewer complex loops compared to cursive.
        block_percentage_loops = block_loop_metrics.get('percentage_shapes_with_loops', 0.0)
        # Score is higher when the percentage of loops is lower.
        block_loop_feature_score = 1.0 - (block_percentage_loops / 100.0)
        block_loop_feature_score = max(0.0, min(1.0, block_loop_feature_score))  # Clamp score [0, 1]

        # --- Combined Block Lettering Style Score ---
        # Weights can be adjusted based on feature importance
        W_BLOCK_ANGULARITY = 1.0
        W_BLOCK_ASPECT_RATIO = 1.0
        W_BLOCK_LOOP = 1.0
        total_block_weight = W_BLOCK_ANGULARITY + W_BLOCK_ASPECT_RATIO + W_BLOCK_LOOP

        block_lettering_style_score = (W_BLOCK_ANGULARITY * block_angularity_feature_score +
                                       W_BLOCK_ASPECT_RATIO * block_aspect_ratio_consistency_score +
                                       W_BLOCK_LOOP * block_loop_feature_score) / total_block_weight


        # =====================================================
        # === SCORE CALCULATION FOR CALLIGRAPHIC HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics
        calligraphic_coverage_metrics = coverage_results.get('metrics', {})
        calligraphic_right_angle_metrics = right_angle_results.get('metrics', {})
        calligraphic_stroke_width_metrics = stroke_width_results.get('metrics', {})

        # --- Continuous Part Coverage Score ---
        # Calligraphy often has continuous strokes, but may have breaks for flourishes.
        calligraphic_coverage_ratio = calligraphic_coverage_metrics.get('continuous_part_coverage_ratio', 0.0)
        # Direct mapping: higher coverage ratio might suggest more connected writing.
        # For calligraphy, this might be nuanced. Let's assume higher is generally better for now.
        calligraphic_coverage_feature_score = max(0.0, min(1.0, calligraphic_coverage_ratio))

        # --- Right Angle Density Score (Inverse) ---
        # Calligraphy tends towards curves rather than sharp right angles.
        calligraphic_right_angle_density = calligraphic_right_angle_metrics.get('right_angle_corner_density', 0.0)
        calligraphic_right_angle_feature_score = 0.0  # Default
        # Threshold: Densities >= this value will get score 0 (less calligraphic)
        CALLIGRAPHIC_RIGHT_ANGLE_DENSITY_THRESHOLD_FOR_MIN_SCORE = 10.0  # Tunable

        if CALLIGRAPHIC_RIGHT_ANGLE_DENSITY_THRESHOLD_FOR_MIN_SCORE > 0:
            # Score decreases linearly as density increases
            calligraphic_right_angle_feature_score = 1.0 - (
                    calligraphic_right_angle_density / CALLIGRAPHIC_RIGHT_ANGLE_DENSITY_THRESHOLD_FOR_MIN_SCORE)
            calligraphic_right_angle_feature_score = max(0.0, min(1.0,
                                                                  calligraphic_right_angle_feature_score))  # Clamp score [0, 1]
        else:
            calligraphic_right_angle_feature_score = 1.0 if calligraphic_right_angle_density == 0 else 0.0

        # --- Stroke Width Variation Score ---
        # Calligraphy is characterized by significant stroke width variation.
        calligraphic_width_variation_coefficient = calligraphic_stroke_width_metrics.get('variation_coefficient', 0.0)
        calligraphic_width_variation_score = 0.0  # Default
        # Threshold for Coefficient of Variation (CV) where the score reaches 1.0 (highly calligraphic). Tunable.
        CALLIGRAPHIC_WIDTH_CV_THRESHOLD_FOR_MAX_SCORE = 0.6  # Higher CV means more variation

        if CALLIGRAPHIC_WIDTH_CV_THRESHOLD_FOR_MAX_SCORE > 0:
            # Linearly scale the score: 0 at CV=0, 1 at CV=THRESHOLD
            calligraphic_width_variation_score = calligraphic_width_variation_coefficient / CALLIGRAPHIC_WIDTH_CV_THRESHOLD_FOR_MAX_SCORE
            calligraphic_width_variation_score = max(0.0,
                                                     min(1.0, calligraphic_width_variation_score))  # Clamp score [0, 1]
        else:
            # Handle edge case of zero threshold
            calligraphic_width_variation_score = 1.0 if calligraphic_width_variation_coefficient > 0 else 0.0

        # --- Combined Calligraphic Style Score ---
        # Weights can be adjusted
        W_CALLIGRAPHIC_COVERAGE = 0.8  # Slightly less weight?
        W_CALLIGRAPHIC_RIGHT_ANGLE = 1.0
        W_CALLIGRAPHIC_WIDTH_VAR = 1.2  # More weight?
        total_calligraphic_weight = W_CALLIGRAPHIC_COVERAGE + W_CALLIGRAPHIC_RIGHT_ANGLE + W_CALLIGRAPHIC_WIDTH_VAR

        calligraphic_style_score = (W_CALLIGRAPHIC_COVERAGE * calligraphic_coverage_feature_score +
                                    W_CALLIGRAPHIC_RIGHT_ANGLE * calligraphic_right_angle_feature_score +
                                    W_CALLIGRAPHIC_WIDTH_VAR * calligraphic_width_variation_score) / total_calligraphic_weight


        # =====================================================
        # === SCORE CALCULATION FOR CURSIVE HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics
        cursive_curvature_metrics = curvature_results.get('metrics', {})
        cursive_loop_metrics = loop_results.get('metrics', {})
        cursive_connectivity_metrics = connectivity_results.get('metrics', {})
        cursive_consistency_metrics = consistency_results.get('metrics', {})

        # --- Curvature/Smoothness Score ---
        # Use average normalized segment length: Smoother cursive -> longer segments.
        cursive_avg_norm_segment_length = cursive_curvature_metrics.get('avg_normalized_segment_length', 0.0)
        # Threshold for avg segment length (fraction of image height) for max score. Tunable.
        CURSIVE_SEGMENT_LENGTH_THRESHOLD_FOR_MAX_SCORE = 0.05
        cursive_curvature_feature_score = 0.0  # Default

        if CURSIVE_SEGMENT_LENGTH_THRESHOLD_FOR_MAX_SCORE > 0:
            # Linearly scale score from 0 up to the threshold.
            cursive_curvature_feature_score = min(1.0, max(0.0,
                                                           cursive_avg_norm_segment_length / CURSIVE_SEGMENT_LENGTH_THRESHOLD_FOR_MAX_SCORE))
        else:
            cursive_curvature_feature_score = 1.0 if cursive_avg_norm_segment_length > 0 else 0.0

        # --- Stroke Connectivity Score ---
        # Lower 'average_components_per_word' means MORE connected (more cursive).
        cursive_avg_components_per_word = cursive_connectivity_metrics.get('average_components_per_word', None)
        cursive_word_count = cursive_connectivity_metrics.get('word_count', 0)
        cursive_connectivity_feature_score = 0.0  # Default
        # Thresholds:
        CURSIVE_CONNECTIVITY_MIN_COMPONENTS_FOR_MAX_SCORE = 1.5  # Below this, score is 1 (highly connected)
        CURSIVE_CONNECTIVITY_MAX_COMPONENTS_FOR_MIN_SCORE = 8.0  # At or above this, score is 0 (very print-like)

        if cursive_word_count > 0 and cursive_avg_components_per_word is not None:
            if cursive_avg_components_per_word <= CURSIVE_CONNECTIVITY_MIN_COMPONENTS_FOR_MAX_SCORE:
                cursive_connectivity_feature_score = 1.0
            elif cursive_avg_components_per_word >= CURSIVE_CONNECTIVITY_MAX_COMPONENTS_FOR_MIN_SCORE:
                cursive_connectivity_feature_score = 0.0
            else:
                # Linear interpolation: score decreases as avg_comps_per_word increases
                cursive_connectivity_feature_score = (
                                                             CURSIVE_CONNECTIVITY_MAX_COMPONENTS_FOR_MIN_SCORE - cursive_avg_components_per_word) / \
                                                     (
                                                             CURSIVE_CONNECTIVITY_MAX_COMPONENTS_FOR_MIN_SCORE - CURSIVE_CONNECTIVITY_MIN_COMPONENTS_FOR_MAX_SCORE)
                cursive_connectivity_feature_score = min(1.0, max(0.0, cursive_connectivity_feature_score))  # Clamp

        # --- Enclosed Loop Ratio Score ---
        # Higher ratio indicates more cursive features (like 'e', 'l', 'o').
        cursive_enclosed_loop_ratio = cursive_loop_metrics.get('enclosed_loop_ratio', 0.0)
        # Direct mapping, potentially scaled if needed, but 0-1 ratio works well.
        cursive_loop_feature_score = min(1.0, max(0.0, cursive_enclosed_loop_ratio))

        # --- Stroke Consistency Score ---
        # More consistent strokes might indicate a more practiced cursive hand.
        cursive_stroke_consistency_index = cursive_consistency_metrics.get('stroke_consistency_index', 0.0)
        # Assuming the index is already normalized between 0 and 1 (1 = most consistent).
        cursive_consistency_feature_score = min(1.0, max(0.0, cursive_stroke_consistency_index))

        # --- Combined Cursive Style Score ---
        W_CURSIVE_CURVATURE = 1.0
        W_CURSIVE_CONNECTIVITY = 1.2  # Connectivity is key for cursive
        W_CURSIVE_LOOP = 0.9
        W_CURSIVE_CONSISTENCY = 0.9
        total_cursive_weight = W_CURSIVE_CURVATURE + W_CURSIVE_CONNECTIVITY + W_CURSIVE_LOOP + W_CURSIVE_CONSISTENCY

        cursive_style_score = (W_CURSIVE_CURVATURE * cursive_curvature_feature_score +
                               W_CURSIVE_CONNECTIVITY * cursive_connectivity_feature_score +
                               W_CURSIVE_LOOP * cursive_loop_feature_score +
                               W_CURSIVE_CONSISTENCY * cursive_consistency_feature_score) / total_cursive_weight


        # =====================================================
        # === SCORE CALCULATION FOR ITALIC HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics
        italic_spacing_metrics = spacing_results.get('metrics', {})
        italic_slant_metrics = slant_angle_results.get('metrics', {})
        italic_vertical_stroke_metrics = vertical_stroke_results.get('metrics', {})

        # --- Letter Spacing Uniformity Score ---
        # Uniform spacing can contribute to a neat italic appearance.
        italic_spacing_is_uniform = italic_spacing_metrics.get('is_uniform')
        if italic_spacing_is_uniform is None:
            # Not enough gaps to determine; assign a neutral score or low confidence?
            italic_spacing_feature_score = 0.5  # Neutral score
        else:
            # Assume uniform spacing is slightly more characteristic of careful italic
            italic_spacing_feature_score = 1.0 if italic_spacing_is_uniform else 0.2  # Penalize non-uniform

        # --- Slant Angle Score ---
        # Significant, consistent slant is the defining feature of italic.
        italic_absolute_vertical_slant = abs(italic_slant_metrics.get('vertical_slant', 0))
        italic_slant_threshold_config = italic_slant_metrics.get('italic_threshold',
                                                                 8)  # Configured threshold from analyzer
        italic_slant_std_dev = italic_slant_metrics.get('slant_std', 0)

        # Score based on angle magnitude relative to threshold
        ITALIC_SLANT_ANGLE_FOR_MAX_SCORE = italic_slant_threshold_config + 10  # e.g., 18 degrees for max score if threshold is 8
        italic_slant_magnitude_score = 0.0
        if ITALIC_SLANT_ANGLE_FOR_MAX_SCORE > 0:
            italic_slant_magnitude_score = min(1.0, max(0.0,
                                                        italic_absolute_vertical_slant / ITALIC_SLANT_ANGLE_FOR_MAX_SCORE))

        # Penalize inconsistency (high standard deviation)
        ITALIC_SLANT_STD_DEV_PENALTY_THRESHOLD = 10.0  # Above this std dev, start penalizing
        slant_consistency_penalty = 0.0
        if italic_slant_std_dev > ITALIC_SLANT_STD_DEV_PENALTY_THRESHOLD:
            # Example penalty: Reduce score by up to 25% for very high std dev
            slant_consistency_penalty = min(0.25, (
                    italic_slant_std_dev - ITALIC_SLANT_STD_DEV_PENALTY_THRESHOLD) / 20.0)  # Tunable

        italic_slant_feature_score = italic_slant_magnitude_score * (1.0 - slant_consistency_penalty)


        # --- Vertical Stroke Proportion Score ---
        # Italic often shows distinct x-height and ascender/descender heights.
        italic_ascender_ratio = italic_vertical_stroke_metrics.get('ascender_ratio', 1.0)  # Default to 1 (uniform)
        italic_vertical_proportion_score = 0.0  # Default
        # Thresholds for scoring based on ratio (higher ratio -> more distinct zones)
        ITALIC_VERTICAL_PROPORTION_BASELINE_RATIO = 1.3  # Ratios below this get low scores
        ITALIC_VERTICAL_PROPORTION_TARGET_RATIO_FOR_MAX_SCORE = 2.5  # Ratios at or above get max score

        if italic_ascender_ratio <= ITALIC_VERTICAL_PROPORTION_BASELINE_RATIO:
            italic_vertical_proportion_score = 0.0
        elif italic_ascender_ratio >= ITALIC_VERTICAL_PROPORTION_TARGET_RATIO_FOR_MAX_SCORE:
            italic_vertical_proportion_score = 1.0
        else:
            # Linear scaling between baseline and target
            italic_vertical_proportion_score = (italic_ascender_ratio - ITALIC_VERTICAL_PROPORTION_BASELINE_RATIO) / \
                                               (
                                                       ITALIC_VERTICAL_PROPORTION_TARGET_RATIO_FOR_MAX_SCORE - ITALIC_VERTICAL_PROPORTION_BASELINE_RATIO)
            italic_vertical_proportion_score = min(1.0, max(0.0, italic_vertical_proportion_score))  # Clamp

        # --- Combined Italic Style Score ---
        W_ITALIC_SPACING = 0.8
        W_ITALIC_SLANT = 1.5  # Slant is most important
        W_ITALIC_VERTICAL = 1.0
        total_italic_weight = W_ITALIC_SPACING + W_ITALIC_SLANT + W_ITALIC_VERTICAL

        italic_style_score = (W_ITALIC_SPACING * italic_spacing_feature_score +
                              W_ITALIC_SLANT * italic_slant_feature_score +
                              W_ITALIC_VERTICAL * italic_vertical_proportion_score) / total_italic_weight


        # =====================================================
        # === SCORE CALCULATION FOR PRINT HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics
        print_vertical_alignment_metrics = vertical_alignment_results.get('metrics', {})
        print_letter_size_metrics = letter_size_results.get('metrics', {})
        print_discrete_letter_metrics = discrete_letter_results.get('metrics', {})

        # --- Vertical Alignment Score ---
        # Print is typically well-aligned vertically. Requires multiple components for meaningful assessment.
        print_component_count_for_align = print_vertical_alignment_metrics.get('component_count', 0)
        print_vertical_alignment_feature_score = 0.0  # Default score if not enough components

        if print_component_count_for_align > 1:
            # Retrieve the individual scores from the analyzer metrics
            print_overall_vertical_alignment_score = print_vertical_alignment_metrics.get('overall_alignment_score',
                                                                                          0.0)

            # Height consistency metric = weighted normalized standard deviation of heights.
            # Higher value means *less* consistency. So, score = 1 - metric_value.
            # Default to 1.0 (max inconsistency) if key is missing.
            print_raw_height_consistency = print_vertical_alignment_metrics.get('height_consistency', 1.0)
            # Calculate the consistency *score* (higher is better)
            print_height_consistency_score = 1.0 - print_raw_height_consistency

            # Combine alignment score (higher=better) and height consistency score (higher=better)
            # Both scores are now expected to be in the range [0, 1]
            calculated_score = (print_overall_vertical_alignment_score + print_height_consistency_score) / 2.0

            # Ensure the combined score is clamped between 0 and 1
            print_vertical_alignment_feature_score = max(0.0, min(1.0, calculated_score))
        else:
            # Assign a low score if there are not enough components (<= 1)
            # because vertical alignment/consistency cannot be reliably measured.
            print_vertical_alignment_feature_score = 0.0

        # --- Letter Size Uniformity Score ---
        # Print often has uniform letter sizes and shapes.
        # Requires at least 2 components (letters) to measure uniformity meaningfully.

        # Get the count of valid components detected by the DiscreteLetterAnalyzer,
        # as these are likely the inputs for the uniformity analyzer.
        num_print_components = print_discrete_letter_metrics.get('num_components', 0)

        print_size_uniformity_feature_score = 0.0  # Default score: Uniformity cannot be assessed

        if num_print_components > 1:
            # Only calculate uniformity if there are multiple components to compare
            print_height_uniformity = print_letter_size_metrics.get('height_uniformity', 0.0)
            print_width_uniformity = print_letter_size_metrics.get('width_uniformity', 0.0)
            print_aspect_ratio_uniformity = print_letter_size_metrics.get('aspect_ratio_uniformity', 0.0)
            print_stroke_width_uniformity = print_letter_size_metrics.get('stroke_width_uniformity', 0.0)

            # Average the uniformity scores. Assume they are all [0, 1] where 1 is max uniformity.
            print_size_uniformity_feature_score = (print_height_uniformity +
                                                   print_width_uniformity +
                                                   print_aspect_ratio_uniformity +
                                                   print_stroke_width_uniformity) / 4.0
            # Clamp the result just in case
            print_size_uniformity_feature_score = max(0.0, min(1.0, print_size_uniformity_feature_score))

        # --- Discrete Letter Score ---
        # Print consists of discrete, separate letters (high number of components relative to strokes/area).
        # Using the discrete letter index (ratio of components to total potential components/area/strokes).
        print_num_components = print_discrete_letter_metrics.get('num_components', 0)
        print_total_components_ref = print_discrete_letter_metrics.get('total_components',
                                                                       0)  # Check what this represents
        # Assuming a higher index/ratio means more discrete letters.
        print_discrete_letter_index = print_num_components / print_total_components_ref if print_total_components_ref > 1 else 0
        print_discrete_letter_feature_score = min(1.0, max(0.0, print_discrete_letter_index))  # Clamp

        # --- Combined Print Style Score ---
        W_PRINT_ALIGNMENT = 1.1
        W_PRINT_SIZE_UNIFORMITY = 1.0
        W_PRINT_DISCRETE = 1.2  # Discreteness is key for print vs cursive
        total_print_weight = W_PRINT_ALIGNMENT + W_PRINT_SIZE_UNIFORMITY + W_PRINT_DISCRETE

        print_style_score = (W_PRINT_ALIGNMENT * print_vertical_alignment_feature_score +
                             W_PRINT_SIZE_UNIFORMITY * print_size_uniformity_feature_score +
                             W_PRINT_DISCRETE * print_discrete_letter_feature_score) / total_print_weight


        # =====================================================
        # === SCORE CALCULATION FOR SHORTHAND HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics
        shorthand_smooth_curves_metrics = smooth_curves_results.get('metrics', {})
        # Use the correctly named results variable from extraction phase
        shorthand_stroke_continuity_metrics = shorthand_continuity_results.get('metrics', {})
        shorthand_symbol_density_metrics = symbol_density_results.get('metrics', {})

        # --- Smooth Curves Score ---
        # Shorthand often involves smooth, flowing curves. Low average angle change indicates smoothness.
        shorthand_avg_abs_angle_change = shorthand_smooth_curves_metrics.get('avg_abs_angle_change',
                                                                             1.0)  # Default assumes unsmooth
        # Threshold for max average angle change (radians) considered "smooth enough". Tunable.
        SHORTHAND_CURVE_SMOOTHNESS_ANGLE_CHANGE_THRESHOLD_RAD = 0.4  # Lower value = stricter smoothness requirement

        # Score = 1 - (value / threshold), max score for 0 change, min score at/above threshold.
        shorthand_curve_smoothness_feature_score = max(0.0, 1.0 - (
                shorthand_avg_abs_angle_change / SHORTHAND_CURVE_SMOOTHNESS_ANGLE_CHANGE_THRESHOLD_RAD))

        # --- Stroke Continuity Score ---
        # Shorthand symbols can be continuous loops or lines, potentially fewer endpoints per component.
        shorthand_num_endpoints = shorthand_stroke_continuity_metrics.get('num_endpoints', 0)
        shorthand_num_components = shorthand_stroke_continuity_metrics.get('num_components', 0)
        shorthand_stroke_continuity_feature_score = 0.0  # Default

        if shorthand_num_components > 0:
            # Ratio of endpoints per component. Lower ratio suggests more continuity (loops=0, lines=2).
            shorthand_endpoint_ratio = shorthand_num_endpoints / shorthand_num_components
            # Threshold for the ratio. Ratios above this are considered discontinuous. Tunable.
            SHORTHAND_STROKE_CONTINUITY_ENDPOINT_RATIO_THRESHOLD = 5.0  # Higher threshold = more lenient

            # Score decreases as ratio increases (less continuous). Max score for ratio 0.
            shorthand_stroke_continuity_feature_score = max(0.0, 1.0 - (
                    shorthand_endpoint_ratio / SHORTHAND_STROKE_CONTINUITY_ENDPOINT_RATIO_THRESHOLD))
        else:
            # No components implies zero continuity score? Or handle as edge case.
            shorthand_stroke_continuity_feature_score = 0.0

        # --- Symbol Density Score ---
        # Shorthand aims to be compact, thus higher density.
        shorthand_density_index = shorthand_symbol_density_metrics.get('density_index', 0.0)
        # Assuming density_index is normalized [0, 1], higher means denser.
        shorthand_symbol_density_feature_score = min(1.0, max(0.0, shorthand_density_index))

        # --- Combined Shorthand Style Score ---
        W_SHORTHAND_SMOOTHNESS = 1.1
        W_SHORTHAND_CONTINUITY = 1.0
        W_SHORTHAND_DENSITY = 1.2  # Density is important for shorthand efficiency
        total_shorthand_weight = W_SHORTHAND_SMOOTHNESS + W_SHORTHAND_CONTINUITY + W_SHORTHAND_DENSITY

        shorthand_style_score = (W_SHORTHAND_SMOOTHNESS * shorthand_curve_smoothness_feature_score +
                                 W_SHORTHAND_CONTINUITY * shorthand_stroke_continuity_feature_score +
                                 W_SHORTHAND_DENSITY * shorthand_symbol_density_feature_score) / total_shorthand_weight

        # =====================================================
        # === FINAL RESPONSE ASSEMBLY ===
        # =====================================================

        # Convert all results to Python native types before sending JSON
        response_data = {
            "processed_image": processed_image_base64,  # Send back the processed image representation
            # Include detailed results from each analyzer
            "analysis_details": {
                "block_lettering": {
                    "angularity": convert_numpy_types(angularity_results),
                    "aspect_ratio": convert_numpy_types(aspect_ratio_results),
                    "loop_detection": convert_numpy_types(loop_detection_results),
                },
                "calligraphic": {
                    "continuous_part_coverage": convert_numpy_types(coverage_results),
                    "right_angle_corner_detection": convert_numpy_types(right_angle_results),
                    "stroke_width_variation": convert_numpy_types(stroke_width_results),
                },
                "cursive": {
                    "stroke_connectivity": convert_numpy_types(connectivity_results),
                    "enclosed_loop_ratio": convert_numpy_types(loop_results),
                    "curvature_continuity": convert_numpy_types(curvature_results),
                    "stroke_consistency": convert_numpy_types(consistency_results),
                },
                "italic": {
                    "vertical_stroke_proportion": convert_numpy_types(vertical_stroke_results),
                    "slant_angle": convert_numpy_types(slant_angle_results),
                    "inter_letter_spacing": convert_numpy_types(spacing_results),
                },
                "print": {
                    "vertical_alignment": convert_numpy_types(vertical_alignment_results),
                    "letter_size_uniformity": convert_numpy_types(letter_size_results),
                    "discrete_letter": convert_numpy_types(discrete_letter_results),
                },
                "shorthand": {
                    # Use the correctly named variable here too
                    "stroke_continuity": convert_numpy_types(shorthand_continuity_results),
                    "smooth_curves": convert_numpy_types(smooth_curves_results),
                    "symbol_density": convert_numpy_types(symbol_density_results),
                }
            },
            # Include the calculated style scores
            "handwriting_style_scores": {
                "block_lettering": {
                    "score": float(block_lettering_style_score),
                    # Optionally include the component scores for debugging/info
                    "component_scores": {
                        "angularity": float(block_angularity_feature_score),
                        "aspect_ratio_consistency": float(block_aspect_ratio_consistency_score),
                        "loop_presence_inverse": float(block_loop_feature_score),
                    }
                },
                "cursive": {
                    "score": float(cursive_style_score),
                    "component_scores": {
                        "curvature_smoothness": float(cursive_curvature_feature_score),
                        "stroke_connectivity": float(cursive_connectivity_feature_score),
                        "enclosed_loop_ratio": float(cursive_loop_feature_score),
                        "stroke_consistency": float(cursive_consistency_feature_score),
                    }
                },
                "calligraphic": {
                    "score": float(calligraphic_style_score),
                    "component_scores": {
                        "coverage": float(calligraphic_coverage_feature_score),
                        "right_angle_inverse": float(calligraphic_right_angle_feature_score),
                        "stroke_width_variation": float(calligraphic_width_variation_score),
                    }
                },
                "italic": {
                    "score": float(italic_style_score),
                    "component_scores": {
                        "spacing_uniformity": float(italic_spacing_feature_score),
                        "slant": float(italic_slant_feature_score),
                        "vertical_proportion": float(italic_vertical_proportion_score),
                    }
                },
                "shorthand": {
                    "score": float(shorthand_style_score),
                    "component_scores": {
                        "curve_smoothness": float(shorthand_curve_smoothness_feature_score),
                        "stroke_continuity": float(shorthand_stroke_continuity_feature_score),
                        "symbol_density": float(shorthand_symbol_density_feature_score),
                    }
                },
                "print": {
                    "score": float(print_style_score),
                    "component_scores": {
                        "vertical_alignment": float(print_vertical_alignment_feature_score),
                        "size_uniformity": float(print_size_uniformity_feature_score),
                        "letter_discreteness": float(print_discrete_letter_feature_score),
                    }
                },
            }
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        # Log the exception for debugging
        import traceback
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
