import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from lib_py.block_lettering.Angularity import AngularityAnalyzer
from lib_py.block_lettering.Aspect_Ratio import AspectRatioAnalyzer
from lib_py.block_lettering.Loop_Detection import LoopDetectionAnalyzer
from lib_py.calligraphic.Continuous_Part_Coverage import ContinuousPartCoverageAnalyzer
from lib_py.calligraphic.Right_Angle_Density import RightAngleAnalyzer
from lib_py.calligraphic.Stroke_Width_Variation import StrokeWidthAnalyzer
from lib_py.cursive.Curvature_Continuity import CursiveCurvatureAnalyzer
from lib_py.cursive.Enclosed_Loop_Ratio import EnclosedLoopAnalyzer
from lib_py.cursive.Stroke_Connectivity_Index import StrokeConnectivityAnalyzer
from lib_py.cursive.Stroke_Consistency import StrokeConsistencyAnalyzer
from lib_py.italic.Inter_Letter_Spacing_Uniformity import LetterSpacingAnalyzer
from lib_py.italic.Slant_Angle import SlantAngleAnalyzer
from lib_py.italic.Vertical_Stroke_Proportion import VerticalStrokeAnalyzer
from lib_py.print.Letter_Discreteness import LetterDiscretenessAnalyzer
from lib_py.print.Letter_Size_Uniformity import LetterUniformityAnalyzer
from lib_py.print.Vertical_Alignment_Consistency import VerticalAlignmentAnalyzer
from lib_py.shorthand.Smooth_Curves import StrokeSmoothnessAnalyzer
from lib_py.shorthand.Stroke_Terminal import StrokeTerminalAnalyzer
from lib_py.shorthand.Symbol_Density import SymbolDensityAnalyzer

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
        # Handle NaN, inf, and -inf values by converting them to None
        if np.isnan(obj) or np.isinf(obj):
            return None
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
        letter_discreteness_analyzer = LetterDiscretenessAnalyzer(request.image, is_base64=True)
        letter_size_analyzer = LetterUniformityAnalyzer(request.image, is_base64=True)
        vertical_alignment_analyzer = VerticalAlignmentAnalyzer(request.image, is_base64=True)
        # Removed: shorthand_continuity_analyzer = StrokeContinuityAnalyzer(request.image, is_base64=True)
        smooth_curves_analyzer = StrokeSmoothnessAnalyzer(request.image, is_base64=True)
        stroke_terminal_analyzer = StrokeTerminalAnalyzer(request.image, is_base64=True)
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
        letter_discreteness_results = letter_discreteness_analyzer.analyze(debug=True)
        letter_size_results = letter_size_analyzer.analyze(debug=True)
        vertical_alignment_results = vertical_alignment_analyzer.analyze(debug=True)
        # Removed: shorthand_continuity_results = shorthand_continuity_analyzer.analyze(debug=True)
        smooth_curves_results = smooth_curves_analyzer.analyze(debug=True)
        stroke_terminal_results = stroke_terminal_analyzer.analyze(debug=True)
        symbol_density_results = symbol_density_analyzer.analyze(debug=True)

        # ===================================================
        # === SCORE CALCULATION FOR BLOCK LETTERING STYLE ===
        # ===================================================
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
                deviation_from_90 = abs(block_median_angle - 90.0)
                block_angularity_feature_score = max(0.0, 1.0 - (deviation_from_90 / 45.0))

        # --- Aspect Ratio Consistency Score ---
        block_aspect_ratio_std_dev = block_aspect_ratio_metrics.get('std_dev_aspect_ratio')
        block_num_letter_candidates = block_aspect_ratio_metrics.get('num_letter_candidates', 0)
        block_aspect_ratio_consistency_score = 0.0  # Default
        BLOCK_AR_STD_DEV_MAX_SCORE_THRESHOLD = 0.15
        BLOCK_AR_STD_DEV_MIN_SCORE_THRESHOLD = 1.0
        if block_num_letter_candidates >= 2 and block_aspect_ratio_std_dev is not None:
            if block_aspect_ratio_std_dev <= BLOCK_AR_STD_DEV_MAX_SCORE_THRESHOLD:
                block_aspect_ratio_consistency_score = 1.0
            elif block_aspect_ratio_std_dev >= BLOCK_AR_STD_DEV_MIN_SCORE_THRESHOLD:
                block_aspect_ratio_consistency_score = 0.0
            else:
                block_aspect_ratio_consistency_score = (
                                                               BLOCK_AR_STD_DEV_MIN_SCORE_THRESHOLD - block_aspect_ratio_std_dev) / \
                                                       (
                                                               BLOCK_AR_STD_DEV_MIN_SCORE_THRESHOLD - BLOCK_AR_STD_DEV_MAX_SCORE_THRESHOLD)
                block_aspect_ratio_consistency_score = max(0.0, min(1.0, block_aspect_ratio_consistency_score))

        # --- Loop Presence Score (Inverse) ---
        block_percentage_loops = block_loop_metrics.get('percentage_shapes_with_loops', 0.0)
        block_loop_feature_score = 1.0 - (block_percentage_loops / 100.0)
        block_loop_feature_score = max(0.0, min(1.0, block_loop_feature_score))

        # --- Combined Block Lettering Style Score ---
        W_BLOCK_ANGULARITY = 1.5
        W_BLOCK_ASPECT_RATIO = 1.0
        W_BLOCK_LOOP = 1.0
        total_block_weight = W_BLOCK_ANGULARITY + W_BLOCK_ASPECT_RATIO + W_BLOCK_LOOP

        # Removed shared_terminal_count check
        if total_block_weight > 0:
            block_lettering_style_score = (W_BLOCK_ANGULARITY * block_angularity_feature_score +
                                           W_BLOCK_ASPECT_RATIO * block_aspect_ratio_consistency_score +
                                           W_BLOCK_LOOP * block_loop_feature_score) / total_block_weight
        else:
            block_lettering_style_score = 0.0

        # ============================================================
        # === SCORE CALCULATION FOR CALLIGRAPHIC HANDWRITING STYLE ===
        # ============================================================
        # Retrieve metrics
        calligraphic_coverage_metrics = coverage_results.get('metrics', {})
        calligraphic_right_angle_metrics = right_angle_results.get('metrics', {})
        calligraphic_stroke_width_metrics = stroke_width_results.get('metrics', {})

        # --- Continuous Part Coverage Score ---
        calligraphic_coverage_ratio = calligraphic_coverage_metrics.get('continuous_part_coverage_ratio', 0.0)
        calligraphic_coverage_feature_score = max(0.0, min(1.0, calligraphic_coverage_ratio))

        # --- Right Angle Density Score (Inverse) ---
        calligraphic_right_angle_density = calligraphic_right_angle_metrics.get('right_angle_corner_density', 0.0)
        calligraphic_right_angle_feature_score = 0.0
        CALLIGRAPHIC_RIGHT_ANGLE_DENSITY_THRESHOLD_FOR_MIN_SCORE = 10.0
        if CALLIGRAPHIC_RIGHT_ANGLE_DENSITY_THRESHOLD_FOR_MIN_SCORE > 0:
            calligraphic_right_angle_feature_score = 1.0 - (
                    calligraphic_right_angle_density / CALLIGRAPHIC_RIGHT_ANGLE_DENSITY_THRESHOLD_FOR_MIN_SCORE)
            calligraphic_right_angle_feature_score = max(0.0, min(1.0,
                                                                  calligraphic_right_angle_feature_score))
        else:
            calligraphic_right_angle_feature_score = 1.0 if calligraphic_right_angle_density == 0 else 0.0

        # --- Stroke Width Variation Score ---
        calligraphic_width_variation_coefficient = calligraphic_stroke_width_metrics.get('variation_coefficient', 0.0)
        calligraphic_width_variation_score = 0.0
        CALLIGRAPHIC_WIDTH_CV_THRESHOLD_FOR_MAX_SCORE = 0.6
        if CALLIGRAPHIC_WIDTH_CV_THRESHOLD_FOR_MAX_SCORE > 0:
            calligraphic_width_variation_score = calligraphic_width_variation_coefficient / CALLIGRAPHIC_WIDTH_CV_THRESHOLD_FOR_MAX_SCORE
            calligraphic_width_variation_score = max(0.0, min(1.0, calligraphic_width_variation_score))
        else:
            calligraphic_width_variation_score = 1.0 if calligraphic_width_variation_coefficient > 0 else 0.0

        # --- Combined Calligraphic Style Score ---
        W_CALLIGRAPHIC_COVERAGE = 0.8
        W_CALLIGRAPHIC_RIGHT_ANGLE = 1.0
        W_CALLIGRAPHIC_WIDTH_VAR = 1.2
        total_calligraphic_weight = W_CALLIGRAPHIC_COVERAGE + W_CALLIGRAPHIC_RIGHT_ANGLE + W_CALLIGRAPHIC_WIDTH_VAR

        # Removed shared_terminal_count check
        if total_calligraphic_weight > 0:
            calligraphic_style_score = (W_CALLIGRAPHIC_COVERAGE * calligraphic_coverage_feature_score +
                                        W_CALLIGRAPHIC_RIGHT_ANGLE * calligraphic_right_angle_feature_score +
                                        W_CALLIGRAPHIC_WIDTH_VAR * calligraphic_width_variation_score) / total_calligraphic_weight
        else:
            calligraphic_style_score = 0.0

        # =======================================================
        # === SCORE CALCULATION FOR CURSIVE HANDWRITING STYLE ===
        # =======================================================
        # Retrieve metrics
        cursive_curvature_metrics = curvature_results.get('metrics', {})
        cursive_loop_metrics = loop_results.get('metrics', {})
        cursive_connectivity_metrics = connectivity_results.get('metrics', {})
        cursive_consistency_metrics = consistency_results.get('metrics', {})

        # --- Curvature Continuity Score ---
        cursive_avg_norm_segment_length = cursive_curvature_metrics.get('avg_normalized_segment_length', 0.0)
        CURSIVE_SEGMENT_LENGTH_THRESHOLD_FOR_MAX_SCORE = 0.05
        cursive_curvature_feature_score = 0.0
        if CURSIVE_SEGMENT_LENGTH_THRESHOLD_FOR_MAX_SCORE > 0:
            cursive_curvature_feature_score = min(1.0, max(0.0,
                                                           cursive_avg_norm_segment_length / CURSIVE_SEGMENT_LENGTH_THRESHOLD_FOR_MAX_SCORE))
        else:
            cursive_curvature_feature_score = 1.0 if cursive_avg_norm_segment_length > 0 else 0.0

        # --- Stroke Connectivity Score ---
        cursive_total_components = cursive_connectivity_metrics.get('total_components', None)
        cursive_bbox_area = cursive_connectivity_metrics.get('bounding_box_area', 0)
        cursive_connectivity_feature_score = 0.0
        CURSIVE_CONNECTIVITY_MAX_COMPONENTS_FOR_MAX_SCORE = 3
        CURSIVE_CONNECTIVITY_MIN_COMPONENTS_FOR_MIN_SCORE = 15
        if cursive_bbox_area > 0 and cursive_total_components is not None and cursive_total_components <= 3:  # Kept check on components for this specific feature logic
            if cursive_total_components <= CURSIVE_CONNECTIVITY_MAX_COMPONENTS_FOR_MAX_SCORE:
                cursive_connectivity_feature_score = 1.0
            elif cursive_total_components >= CURSIVE_CONNECTIVITY_MIN_COMPONENTS_FOR_MIN_SCORE:
                cursive_connectivity_feature_score = 0.0
            else:
                denominator = (
                        CURSIVE_CONNECTIVITY_MIN_COMPONENTS_FOR_MIN_SCORE - CURSIVE_CONNECTIVITY_MAX_COMPONENTS_FOR_MAX_SCORE)
                if denominator > 0:
                    cursive_connectivity_feature_score = (
                                                                 CURSIVE_CONNECTIVITY_MIN_COMPONENTS_FOR_MIN_SCORE - cursive_total_components) / denominator
                    cursive_connectivity_feature_score = min(1.0, max(0.0, cursive_connectivity_feature_score))
                else:
                    cursive_connectivity_feature_score = 0.0

        # --- Enclosed Loop Ratio Score ---
        cursive_enclosed_loop_ratio = cursive_loop_metrics.get('avg_word_loopiness', 0.0)
        cursive_loop_feature_score = min(1.0, max(0.0, cursive_enclosed_loop_ratio))

        # --- Stroke Consistency Score ---
        cursive_stroke_consistency_index = cursive_consistency_metrics.get('stroke_consistency_index', 0.0)
        cursive_consistency_feature_score = min(1.0, max(0.0, cursive_stroke_consistency_index))

        # --- Combined Cursive Style Score ---
        W_CURSIVE_CURVATURE = 1.0
        W_CURSIVE_CONNECTIVITY = 1.2
        W_CURSIVE_LOOP = 0.9
        W_CURSIVE_CONSISTENCY = 0.9
        total_cursive_weight = W_CURSIVE_CURVATURE + W_CURSIVE_CONNECTIVITY + W_CURSIVE_LOOP + W_CURSIVE_CONSISTENCY

        # Removed shared_terminal_count check, kept other conditions specific to cursive features
        if total_cursive_weight > 0 and cursive_total_components is not None and cursive_total_components <= 3:
            cursive_style_score = (W_CURSIVE_CURVATURE * cursive_curvature_feature_score +
                                   W_CURSIVE_CONNECTIVITY * cursive_connectivity_feature_score +
                                   W_CURSIVE_LOOP * cursive_loop_feature_score +
                                   W_CURSIVE_CONSISTENCY * cursive_consistency_feature_score) / total_cursive_weight
        else:
            cursive_style_score = 0.0

        # Clamp final score
        cursive_style_score = max(0.0, min(1.0, cursive_style_score))

        # ======================================================
        # === SCORE CALCULATION FOR ITALIC HANDWRITING STYLE ===
        # ======================================================
        # Retrieve metrics
        italic_spacing_metrics = spacing_results.get('metrics', {})
        italic_slant_metrics = slant_angle_results.get('metrics', {})
        italic_vertical_stroke_metrics = vertical_stroke_results.get('metrics', {})

        # --- Letter Spacing Uniformity Score ---
        # Updated logic using metrics from LetterSpacingAnalyzer
        valley_count = italic_spacing_metrics.get('valley_count', 0)
        avg_valley_width = italic_spacing_metrics.get('avg_valley_width', 0.0)
        valley_width_std = italic_spacing_metrics.get('valley_width_std', 0.0)

        italic_spacing_feature_score = 0.0  # Default score

        MIN_VALLEYS_FOR_STD_SCORE = 2  # Need at least 2 gaps to measure variation reliably
        # Define thresholds for Coefficient of Variation (CV = std_dev / mean)
        # Lower CV means more uniform spacing (good). These thresholds might need tuning.
        MAX_CV_FOR_PERFECT_SCORE = 0.30  # Below this CV -> score 1.0
        MIN_CV_FOR_ZERO_SCORE = 0.80  # Above this CV -> score 0.0

        if valley_count >= MIN_VALLEYS_FOR_STD_SCORE:
            # Calculate Coefficient of Variation (CV) for valley widths
            if avg_valley_width > 1e-6:  # Avoid division by zero or near-zero average
                valley_cv = valley_width_std / avg_valley_width

                if valley_cv <= MAX_CV_FOR_PERFECT_SCORE:
                    italic_spacing_feature_score = 1.0
                elif valley_cv >= MIN_CV_FOR_ZERO_SCORE:
                    italic_spacing_feature_score = 0.0
                else:
                    # Linear interpolation between the thresholds
                    score_range = MIN_CV_FOR_ZERO_SCORE - MAX_CV_FOR_PERFECT_SCORE
                    if score_range > 1e-6:  # Avoid division by zero if thresholds are the same
                        # Score decreases as CV increases
                        italic_spacing_feature_score = 1.0 - (valley_cv - MAX_CV_FOR_PERFECT_SCORE) / score_range
                    else:  # Should ideally not happen if thresholds are set reasonably
                        italic_spacing_feature_score = 0.5  # Assign mid-point if thresholds somehow collapse
            else:
                # Sufficient valleys detected, but average width is effectively zero?
                # This is unusual and likely indicates poor segmentation or connected script. Treat as non-uniform.
                italic_spacing_feature_score = 0.0

        elif valley_count == 1:
            # Only one gap detected - hard to judge uniformity. Assign a low score.
            italic_spacing_feature_score = 0.1
        # else valley_count == 0: score remains 0.0 (default) - no inter-letter gaps found.

        # Final clamping to ensure score is strictly within [0, 1]
        italic_spacing_feature_score = max(0.0, min(1.0, italic_spacing_feature_score))

        # --- Slant Angle Score ---
        italic_is_slant_detected = italic_slant_metrics.get('is_italic', False)
        italic_vertical_slant = italic_slant_metrics.get('vertical_slant', 0.0)
        italic_slant_threshold_config = italic_slant_metrics.get('italic_threshold', 8.0)
        italic_slant_std_dev = italic_slant_metrics.get('slant_std', 0.0)
        italic_slant_feature_score = 0.0

        if italic_is_slant_detected:
            slant_excess = abs(italic_vertical_slant) - italic_slant_threshold_config
            TARGET_EXCESS_FOR_MAX_SCORE = 10.0
            BASE_SCORE_ON_THRESHOLD = 0.5
            magnitude_component = min(1.0, max(0.0,
                                               slant_excess / TARGET_EXCESS_FOR_MAX_SCORE if TARGET_EXCESS_FOR_MAX_SCORE > 0 else 0.0))
            magnitude_score = BASE_SCORE_ON_THRESHOLD + (1.0 - BASE_SCORE_ON_THRESHOLD) * magnitude_component

            ITALIC_SLANT_STD_DEV_PENALTY_THRESHOLD = 10.0
            MAX_CONSISTENCY_PENALTY = 0.30
            slant_consistency_penalty = 0.0
            if italic_slant_std_dev > ITALIC_SLANT_STD_DEV_PENALTY_THRESHOLD:
                penalty_factor = (italic_slant_std_dev - ITALIC_SLANT_STD_DEV_PENALTY_THRESHOLD) / (
                        2 * ITALIC_SLANT_STD_DEV_PENALTY_THRESHOLD)
                slant_consistency_penalty = min(MAX_CONSISTENCY_PENALTY, max(0.0, penalty_factor))

            italic_slant_feature_score = magnitude_score * (1.0 - slant_consistency_penalty)
            italic_slant_feature_score = max(0.0, min(1.0, italic_slant_feature_score))

        # --- Vertical Stroke Proportion Score ---
        italic_ascender_ratio = italic_vertical_stroke_metrics.get('ascender_ratio', 1.0)
        italic_vertical_proportion_score = 0.0
        ITALIC_VERTICAL_PROPORTION_BASELINE_RATIO = 1.2
        ITALIC_VERTICAL_PROPORTION_TARGET_RATIO_FOR_MAX_SCORE = 2.5

        if italic_ascender_ratio <= ITALIC_VERTICAL_PROPORTION_BASELINE_RATIO:
            italic_vertical_proportion_score = 0.0
        elif italic_ascender_ratio >= ITALIC_VERTICAL_PROPORTION_TARGET_RATIO_FOR_MAX_SCORE:
            italic_vertical_proportion_score = 1.0
        else:
            italic_vertical_proportion_score = (italic_ascender_ratio - ITALIC_VERTICAL_PROPORTION_BASELINE_RATIO) / \
                                               (
                                                       ITALIC_VERTICAL_PROPORTION_TARGET_RATIO_FOR_MAX_SCORE - ITALIC_VERTICAL_PROPORTION_BASELINE_RATIO)
            italic_vertical_proportion_score = min(1.0, max(0.0, italic_vertical_proportion_score))

        # --- Combined Italic Style Score ---
        W_ITALIC_SPACING = 0.8
        W_ITALIC_SLANT = 1.2
        W_ITALIC_VERTICAL = 1.0
        total_italic_weight = W_ITALIC_SPACING + W_ITALIC_SLANT + W_ITALIC_VERTICAL

        # Removed shared_terminal_count check, kept other condition specific to italic features
        if total_italic_weight > 0 and italic_is_slant_detected:
            italic_style_score = (W_ITALIC_SPACING * italic_spacing_feature_score +
                                  W_ITALIC_SLANT * italic_slant_feature_score +
                                  W_ITALIC_VERTICAL * italic_vertical_proportion_score) / total_italic_weight
        else:
            italic_style_score = 0.0

        # =====================================================
        # === SCORE CALCULATION FOR PRINT HANDWRITING STYLE ===
        # =====================================================
        # Retrieve metrics
        print_vertical_alignment_metrics = vertical_alignment_results.get('metrics', {})
        print_letter_size_metrics = letter_size_results.get('metrics', {})
        print_letter_discreteness_metrics = letter_discreteness_results.get('metrics', {})

        # --- Vertical Alignment Score ---
        print_component_count_for_align = print_vertical_alignment_metrics.get('component_count', 0)
        print_vertical_alignment_feature_score = 0.0

        if print_component_count_for_align > 1:
            print_overall_vertical_alignment_score = print_vertical_alignment_metrics.get('overall_alignment_score',
                                                                                          0.0)
            print_raw_height_consistency = print_vertical_alignment_metrics.get('height_consistency', 1.0)
            print_height_consistency_score = 1.0 - print_raw_height_consistency
            calculated_score = (print_overall_vertical_alignment_score + print_height_consistency_score) / 2.0
            print_vertical_alignment_feature_score = max(0.0, min(1.0, calculated_score))
        else:
            print_vertical_alignment_feature_score = 0.0

        # --- Letter Size Uniformity Score ---
        num_print_components = print_letter_discreteness_metrics.get('num_components', 0)
        print_size_uniformity_feature_score = 0.0

        if num_print_components > 1:
            print_height_uniformity = print_letter_size_metrics.get('height_uniformity', 0.0)
            print_width_uniformity = print_letter_size_metrics.get('width_uniformity', 0.0)
            print_aspect_ratio_uniformity = print_letter_size_metrics.get('aspect_ratio_uniformity', 0.0)
            print_stroke_width_uniformity = print_letter_size_metrics.get('stroke_width_uniformity', 0.0)
            print_size_uniformity_feature_score = (print_height_uniformity +
                                                   print_width_uniformity +
                                                   print_aspect_ratio_uniformity +
                                                   print_stroke_width_uniformity) / 4.0
            print_size_uniformity_feature_score = max(0.0, min(1.0, print_size_uniformity_feature_score))

        # --- Letter Discreteness Score ---
        is_continuous = print_letter_discreteness_metrics.get('is_likely_continuous', False)
        num_spaces = print_letter_discreteness_metrics.get('num_inter_letter_spaces', 0)
        cv_space = print_letter_discreteness_metrics.get('cv_inter_letter_space', 1.0)
        print_letter_discreteness_feature_score = 0.0

        if is_continuous:
            print_letter_discreteness_feature_score = 0.0
        elif num_spaces <= 1:
            print_letter_discreteness_feature_score = 0.0
        else:
            print_letter_discreteness_feature_score = max(0.0, 1.0 - cv_space / 2.0)

        # --- Combined Print Style Score ---
        W_PRINT_ALIGNMENT = 1.1
        W_PRINT_SIZE_UNIFORMITY = 1.0
        W_PRINT_DISCRETE = 1.2
        total_print_weight = W_PRINT_ALIGNMENT + W_PRINT_SIZE_UNIFORMITY + W_PRINT_DISCRETE

        # Removed shared_terminal_count check
        if total_print_weight > 0:
            print_style_score = (W_PRINT_ALIGNMENT * print_vertical_alignment_feature_score +
                                 W_PRINT_SIZE_UNIFORMITY * print_size_uniformity_feature_score +
                                 W_PRINT_DISCRETE * print_letter_discreteness_feature_score) / total_print_weight
        else:
            print_style_score = 0.0

        # =========================================================
        # === SCORE CALCULATION FOR SHORTHAND HANDWRITING STYLE ===
        # =========================================================
        # Retrieve metrics
        shorthand_smooth_curves_metrics = smooth_curves_results.get('metrics', {})
        # Removed: shorthand_stroke_continuity_metrics retrieval
        shorthand_stroke_terminal_metrics = stroke_terminal_results.get('metrics', {})
        shorthand_symbol_density_metrics = symbol_density_results.get('metrics', {})

        # --- Smooth Curves Score ---
        shorthand_avg_abs_angle_change = shorthand_smooth_curves_metrics.get('avg_abs_angle_change', 1.0)
        SHORTHAND_CURVE_SMOOTHNESS_ANGLE_CHANGE_THRESHOLD_RAD = 0.4
        shorthand_curve_smoothness_feature_score = max(0.0, 1.0 - (
                shorthand_avg_abs_angle_change / SHORTHAND_CURVE_SMOOTHNESS_ANGLE_CHANGE_THRESHOLD_RAD))

        # --- Stroke Continuity Score --- (REMOVED)
        # Removed: Calculation of shorthand_stroke_continuity_feature_score

        # --- Stroke Terminal Score ---
        shorthand_terminal_count = shorthand_stroke_terminal_metrics.get('terminal_count', 1)

        MIN_TERMINAL_COUNT = 1
        MAX_TERMINAL_COUNT = 10

        shorthand_stroke_terminal_feature_score = 0.0

        if MIN_TERMINAL_COUNT <= shorthand_terminal_count <= MAX_TERMINAL_COUNT:
            shorthand_stroke_terminal_feature_score = (
                                                              MAX_TERMINAL_COUNT - shorthand_terminal_count) / MAX_TERMINAL_COUNT
            shorthand_stroke_terminal_feature_score = min(1.0, max(0.0, 1.5 * shorthand_stroke_terminal_feature_score))
        else:
            shorthand_stroke_terminal_feature_score = 0.0

        shorthand_stroke_terminal_feature_score = max(0.0, min(1.0, shorthand_stroke_terminal_feature_score))

        # --- Symbol Density Score ---
        shorthand_density = shorthand_symbol_density_metrics.get('symbol_density', 0.0)

        shorthand_symbol_density_feature_score = 0.0

        # --- Define Density Thresholds ---
        # Below this, likely a blank image or just noise -> score 0
        BLANK_THRESHOLD = 0.005
        # Below this (but >= BLANK_THRESHOLD), density is low enough for a perfect score -> score 1.0
        PERFECT_SCORE_THRESHOLD = 0.025
        # Above this, density is too high for shorthand style -> score 0
        MAX_DENSITY = 0.20  # You can adjust this upper limit based on experiments

        # --- Calculate Score Based on Density Ranges ---
        if shorthand_density < BLANK_THRESHOLD:
            # Density is extremely low, likely indicating a blank image or insignificant noise.
            shorthand_symbol_density_feature_score = 0.0

        elif shorthand_density < PERFECT_SCORE_THRESHOLD:
            # Density is low and within the ideal range for shorthand sparsity.
            shorthand_symbol_density_feature_score = 1.0

        elif shorthand_density <= MAX_DENSITY:
            # Density is between the perfect threshold and the maximum allowed.
            # Score decreases linearly from 1.0 down to 0.0 across this range.
            # Calculate the span of this scoring range:
            relevant_density_range = MAX_DENSITY - PERFECT_SCORE_THRESHOLD

            if relevant_density_range > 1e-6:  # Avoid division by zero if thresholds are the same
                # Calculate how far the current density is into this range (0 at the start, 1 at the end)
                density_position = (shorthand_density - PERFECT_SCORE_THRESHOLD) / relevant_density_range
                # Invert the position to get the score (1.0 at the start, 0.0 at the end)
                shorthand_symbol_density_feature_score = 1.0 - density_position
            else:
                # Edge case: PERFECT_SCORE_THRESHOLD == MAX_DENSITY
                # Since we passed the previous elif, density must be exactly at this threshold.
                # Assign 1.0 as it met the lower bound of acceptable density.
                shorthand_symbol_density_feature_score = 1.0

        else:  # shorthand_density > MAX_DENSITY
            # Density exceeds the maximum acceptable level for this style.
            shorthand_symbol_density_feature_score = 0.0

        # --- Final Clamping (Safety Net) ---
        # Ensure the score strictly stays within [0.0, 1.0] due to potential floating-point nuances.
        shorthand_symbol_density_feature_score = max(0.0, min(1.0, shorthand_symbol_density_feature_score))

        # --- Combined Shorthand Style Score ---
        W_SHORTHAND_SMOOTHNESS = 0.7
        # Removed: W_SHORTHAND_CONTINUITY = 1.0
        W_SHORTHAND_TERMINAL = 1.15
        W_SHORTHAND_DENSITY = 1.15

        # Updated: total_shorthand_weight calculation
        total_shorthand_weight = W_SHORTHAND_SMOOTHNESS + W_SHORTHAND_TERMINAL + W_SHORTHAND_DENSITY

        # Removed checks involving shorthand_num_components and shared_terminal_count
        if total_shorthand_weight > 0:
            # Updated: Shorthand score calculation (removed continuity term)
            shorthand_style_score = (W_SHORTHAND_SMOOTHNESS * shorthand_curve_smoothness_feature_score +
                                     W_SHORTHAND_TERMINAL * shorthand_stroke_terminal_feature_score +
                                     W_SHORTHAND_DENSITY * shorthand_symbol_density_feature_score) / total_shorthand_weight
        else:
            shorthand_style_score = 0.0

        # ==============================================================================
        # === SCORE CORRECTION WITH KEY FEATURE (ADD IF NEEDED NECESSARY CORRECTION) ===
        # ==============================================================================
        # If the letters in the word is not angular enough, it is considered not blocky
        if block_angularity_feature_score < 0.45:
            block_lettering_style_score = 0.0
        # As shorthand has all the features similar to cursive and calligraphic,
        # the stroke terminal count of the handwriting style will finally determine if it is shorthand or not.
        # If the count of the terminal in the word is <= 5,
        # it is considered shorthand as shorthand won't have too much terminal points
        # and cursive as well as calligraphic, will have many terminal points in the word in the word in majority cases.
        if shorthand_terminal_count <= 5:
            cursive_style_score = 0.0
            calligraphic_style_score = 0.0

        # If the handwriting style is italic, return 0 score for all other styles
        if italic_is_slant_detected:
            block_lettering_style_score = 0.0
            calligraphic_style_score = 0.0
            cursive_style_score = 0.0
            shorthand_style_score = 0.0
            print_style_score = 0.0

        # ===============================
        # === FINAL RESPONSE ASSEMBLY ===
        # ===============================
        response_data = \
            {
                # Include detailed results from each analyzer
                "analysis_details": {
                    "block_lettering": {
                        "angularity": {
                            "data": convert_numpy_types(angularity_results),
                            "is_dominant": False,
                            "is_shared": True,
                            "weightage": W_BLOCK_ANGULARITY
                        },
                        "aspect_ratio": {
                            "data": convert_numpy_types(aspect_ratio_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_BLOCK_ASPECT_RATIO
                        },
                        "loop_detection": {
                            "data": convert_numpy_types(loop_detection_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_BLOCK_LOOP
                        }
                    },
                    "calligraphic": {
                        "continuous_part_coverage": {
                            "data": convert_numpy_types(coverage_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_CALLIGRAPHIC_COVERAGE
                        },
                        "right_angle_corner_detection": {
                            "data": convert_numpy_types(right_angle_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_CALLIGRAPHIC_RIGHT_ANGLE
                        },
                        "stroke_width_variation": {
                            "data": convert_numpy_types(stroke_width_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_CALLIGRAPHIC_WIDTH_VAR
                        }
                    },
                    "cursive": {
                        "stroke_connectivity": {
                            "data": convert_numpy_types(connectivity_results),
                            "is_dominant": True,
                            "is_shared": False,
                            "weightage": W_CURSIVE_CONNECTIVITY
                        },
                        "enclosed_loop_ratio": {
                            "data": convert_numpy_types(loop_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_CURSIVE_LOOP
                        },
                        "curvature_continuity": {
                            "data": convert_numpy_types(curvature_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_CURSIVE_CURVATURE
                        },
                        "stroke_consistency": {
                            "data": convert_numpy_types(consistency_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_CURSIVE_CONSISTENCY
                        }
                    },
                    "italic": {
                        "slant_angle": {
                            "data": convert_numpy_types(slant_angle_results),
                            "is_dominant": True,
                            "is_shared": True,
                            "weightage": W_ITALIC_SLANT
                        },
                        "vertical_stroke_proportion": {
                            "data": convert_numpy_types(vertical_stroke_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_ITALIC_VERTICAL
                        },
                        "inter_letter_spacing": {
                            "data": convert_numpy_types(spacing_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_ITALIC_SPACING
                        }
                    },
                    "print": {
                        "vertical_alignment": {
                            "data": convert_numpy_types(vertical_alignment_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_PRINT_ALIGNMENT
                        },
                        "letter_size_uniformity": {
                            "data": convert_numpy_types(letter_size_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_PRINT_SIZE_UNIFORMITY
                        },
                        "discrete_letter": {
                            "data": convert_numpy_types(letter_discreteness_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_PRINT_DISCRETE
                        }
                    },
                    "shorthand": {
                        "stroke_terminal": {
                            "data": convert_numpy_types(stroke_terminal_results),
                            "is_dominant": False,
                            "is_shared": True,
                            "weightage": W_SHORTHAND_TERMINAL
                        },
                        "curve_smoothness": {
                            "data": convert_numpy_types(smooth_curves_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_SHORTHAND_SMOOTHNESS
                        },
                        "symbol_density": {
                            "data": convert_numpy_types(symbol_density_results),
                            "is_dominant": False,
                            "is_shared": False,
                            "weightage": W_SHORTHAND_DENSITY
                        }
                    }
                },
                # Include the calculated style scores
                "handwriting_style_scores": {
                    "block_lettering": {
                        "score": float(block_lettering_style_score),
                        "component_scores": {
                            "angularity": float(block_angularity_feature_score),
                            "aspect_ratio_consistency": float(block_aspect_ratio_consistency_score),
                            "loop_presence_inverse": float(block_loop_feature_score),
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
                    "cursive": {
                        "score": float(cursive_style_score),
                        "component_scores": {
                            "curvature_continuity": float(cursive_curvature_feature_score),
                            "stroke_connectivity": float(cursive_connectivity_feature_score),
                            "enclosed_loop_ratio": float(cursive_loop_feature_score),
                            "stroke_consistency": float(cursive_consistency_feature_score),
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
                    "print": {
                        "score": float(print_style_score),
                        "component_scores": {
                            "vertical_alignment": float(print_vertical_alignment_feature_score),
                            "size_uniformity": float(print_size_uniformity_feature_score),
                            "letter_discreteness": float(print_letter_discreteness_feature_score),
                        }
                    },
                    "shorthand": {
                        "score": float(shorthand_style_score),
                        "component_scores": {
                            "curve_smoothness": float(shorthand_curve_smoothness_feature_score),
                            # Removed: "stroke_continuity": float(shorthand_stroke_continuity_feature_score),
                            "stroke_terminal": float(shorthand_stroke_terminal_feature_score),
                            "symbol_density": float(shorthand_symbol_density_feature_score)
                        }
                    },
                }
            }
        # Convert all results to Python native types before sending JSON

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
