from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

from helper import preprocess_image
from lib_py.block_lettering.angularity import BlockLetterAnalyzer
from lib_py.block_lettering.pen_pressure import PenPressureAnalyzer
from lib_py.block_lettering.uppercase_ratio import UppercaseRatioAnalyzer
from lib_py.calligraphic.artistics_consistency import CalligraphicAnalyzer
from lib_py.calligraphic.flourish_extension_ratio import FlourishAnalyzer
from lib_py.calligraphic.stroke_width_variation import StrokeWidthAnalyzer
from lib_py.cursive.Curvature_Continuity import CursiveCurvatureAnalyzer
from lib_py.cursive.Enclosed_Loop_Ratio import EnclosedLoopAnalyzer
from lib_py.cursive.Stroke_Connectivity_Index import StrokeConnectivityAnalyzer
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
        angularity_analyzer = BlockLetterAnalyzer(request.image, is_base64=True)
        angularity_results = angularity_analyzer.analyze(debug=True)

        pen_pressure_analyzer = PenPressureAnalyzer(request.image, is_base64=True)
        pen_pressure_results = pen_pressure_analyzer.analyze(debug=True)

        uppercase_analyzer = UppercaseRatioAnalyzer(request.image, is_base64=True)
        uppercase_results = uppercase_analyzer.analyze(debug=True)

        # Calligraphic Analysis
        artistic_analyzer = CalligraphicAnalyzer(request.image, is_base64=True)
        artistic_results = artistic_analyzer.analyze(debug=True)

        flourish_analyzer = FlourishAnalyzer(request.image, is_base64=True)
        flourish_results = flourish_analyzer.analyze(debug=True)

        stroke_width_analyzer = StrokeWidthAnalyzer(request.image, is_base64=True)
        stroke_width_results = stroke_width_analyzer.analyze(debug=True)

        # Cursive Analysis
        curvature_analyzer = CursiveCurvatureAnalyzer(request.image, is_base64=True)
        curvature_results = curvature_analyzer.analyze(debug=True)

        loop_analyzer = EnclosedLoopAnalyzer(request.image, is_base64=True)
        loop_results = loop_analyzer.analyze(debug=True)
        
        connectivity_analyzer = StrokeConnectivityAnalyzer(request.image, is_base64=True)
        connectivity_results = connectivity_analyzer.analyze(debug=True)

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

        # --- Score Calculation for Block Lettering --- #
        # Retrieve metrics from the analyzers
        angularity_metrics = angularity_results.get('metrics', {})
        pen_pressure_metrics = pen_pressure_results.get('metrics', {})
        uppercase_metrics = uppercase_results.get('metrics', {})

        # Angularity: lower average deviation indicates sharper (block) corners.
        if angularity_metrics.get('shape_count', 0) > 0:
            avg_deviation = angularity_metrics.get('avg_deviation', 0)
            # Normalize using a threshold (45°): deviation 0 gives score 1; 45° or more gives score 0.
            angularity_score = max(0, 1 - (avg_deviation / 45))
        else:
            angularity_score = 0

        # Pen Pressure: lower coefficient of variation implies more uniform strokes.
        coefficient_of_variation = pen_pressure_metrics.get('coefficient_of_variation', 1)
        # For a perfect uniform stroke, CV=0 (score 1). If CV>=1, score is 0.
        pen_pressure_score = max(0, min(1 - coefficient_of_variation, 1))

        # Uppercase Ratio: higher ratio (closer to 1) suggests block lettering.
        uppercase_ratio = uppercase_metrics.get('uppercase_ratio', 0)
        uppercase_score = uppercase_ratio

        # Combined block lettering score (equal weights)
        block_lettering_score = (angularity_score + pen_pressure_score + uppercase_score) / 3

        # --- Score Calculation for Calligraphic --- #
        # Retrieve metrics from the analyzers
        artistic_metrics = artistic_results.get('metrics', {})
        flourish_metrics = flourish_results.get('metrics', {})
        stroke_width_metrics = stroke_width_results.get('metrics', {})

        # Artistic consistency: higher consistency suggests calligraphic training
        artistic_consistency = artistic_metrics.get('artistic_index', 0)
        artistic_score = min(1, max(0, artistic_consistency))

        # Flourish extension: higher ratio suggests more calligraphic features
        flourish_ratio = flourish_metrics.get('flourish_ratio', 0)
        flourish_score = min(1, max(0, flourish_ratio))
        
        # Stroke width variation: higher variation suggests calligraphic style
        width_variation = stroke_width_metrics.get('width_variation_index', 0)
        width_score = min(1, max(0, width_variation))

        # Combined calligraphic score (equal weights)
        calligraphic_score = (width_score + flourish_score + artistic_score) / 3

        # --- Score Calculation for Cursive --- #
        # Retrieve metrics from the analyzers
        curvature_metrics = curvature_results.get('metrics', {})
        loop_metrics = loop_results.get('metrics', {})
        connectivity_metrics = connectivity_results.get('metrics', {})

        # Curvature continuity: higher continuity indicates cursive writing
        curvature_continuity = curvature_metrics.get('curvature_index', 0)
        curvature_score = min(1, max(0, curvature_continuity))
        
        # Stroke connectivity: higher connectivity index indicates more cursive writing
        connectivity_index = connectivity_metrics.get('connectivity_index', 0)
        connectivity_score = min(1, max(0, connectivity_index))

        # Enclosed loop ratio: higher ratio indicates more cursive features
        loop_ratio = loop_metrics.get('enclosed_loop_ratio', 0)
        loop_score = min(1, max(0, loop_ratio))

        # Combined cursive score (equal weights)
        cursive_score = (connectivity_score + loop_score + curvature_score) / 3

         # --- Score Calculation for Italic --- #
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
        
        # Vertical stroke proportion: lower proportion suggests more italic style
        vertical_proportion = vertical_stroke_metrics.get('vertical_proportion', 1)
        vertical_score = min(1, max(0, 1 - vertical_proportion))
        
        # Combined italic score (equal weights)
        italic_score = (vertical_score + slant_score + spacing_score) / 3

        # --- Score Calculation for Print --- #
        # Retrieve metrics from the analyzers
        vertical_alignment_metrics = vertical_alignment_results.get('metrics', {})
        letter_size_metrics = letter_size_results.get('metrics', {})
        discrete_letter_metrics = discrete_letter_results.get('metrics', {})

        # Vertical alignment: higher alignment indicates print writing
        alignment_index = vertical_alignment_metrics.get('alignment_index', 0)
        alignment_score = min(1, max(0, alignment_index))

        # Letter size uniformity: higher uniformity indicates print
        size_uniformity = letter_size_metrics.get('size_uniformity', 0)
        size_score = min(1, max(0, size_uniformity))

        # Discrete letters: higher discreteness indicates print over cursive
        letter_discreteness = discrete_letter_metrics.get('discrete_index', 0)
        discrete_score = min(1, max(0, letter_discreteness))

        # Combined print score (equal weights)
        print_score = (alignment_score + size_score + discrete_score) / 3

        # --- Score Calculation for Shorthand --- #
        # Retrieve metrics from the analyzers
        smooth_curves_metrics = smooth_curves_results.get('metrics', {})
        continuity_metrics = continuity_results.get('metrics', {})
        symbol_density_metrics = symbol_density_results.get('metrics', {}) 

        # Smooth curves: higher smoothness indicates shorthand writing
        curve_smoothness = smooth_curves_metrics.get('smoothness_index', 0)
        curve_score = min(1, max(0, curve_smoothness))

        # Stroke continuity: higher continuity indicates shorthand
        continuity_index = continuity_metrics.get('continuity_index', 0)
        continuity_score = min(1, max(0, continuity_index))

        # Symbol density: higher density indicates shorthand
        density_index = symbol_density_metrics.get('density_index', 0)
        density_score = min(1, max(0, density_index)) 
        
        # Combined shorthand score (equal weights)
        shorthand_score = (continuity_score + curve_score + density_score) / 3 
        

        # Convert all results to Python native types
        response = {
            "processed_image": processed_image,
            # Block lettering
            "angularity": convert_numpy_types(angularity_results),
            "uppercase_ratio": convert_numpy_types(uppercase_results),
            "pen_pressure": convert_numpy_types(pen_pressure_results),

            # calligraphic"
            "stroke_width_variation": convert_numpy_types(stroke_width_results),
            "flourish_extension": convert_numpy_types(flourish_results),
            "artistic_consistency": convert_numpy_types(artistic_results),

            # cursive"
            "stroke_connectivity": convert_numpy_types(connectivity_results),
            "enclosed_loop_ratio": convert_numpy_types(loop_results),
            "curvature_continuity": convert_numpy_types(curvature_results),

            # italic"
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
