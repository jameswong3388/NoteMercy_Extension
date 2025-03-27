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

        uppercase_analyzer = UppercaseRatioAnalyzer(request.image, is_base64=True)
        uppercase_results = uppercase_analyzer.analyze(debug=True)

        pen_pressure_analyzer = PenPressureAnalyzer(request.image, is_base64=True)
        pen_pressure_results = pen_pressure_analyzer.analyze(debug=True)

        # # Italic Analysis
        vertical_stroke_analyzer = VerticalStrokeAnalyzer(request.image, is_base64=True)
        vertical_stroke_results = vertical_stroke_analyzer.analyze(debug=True)

        slant_angle_analyzer = SlantAngleAnalyzer(request.image, is_base64=True)
        slant_angle_results = slant_angle_analyzer.analyze(debug=True)

        spacing_analyzer = LetterSpacingAnalyzer(request.image, is_base64=True)
        spacing_results = spacing_analyzer.analyze(debug=True)

        # Cursive Analysis
        connectivity_analyzer = StrokeConnectivityAnalyzer(request.image, is_base64=True)
        connectivity_results = connectivity_analyzer.analyze(debug=True)

        loop_analyzer = EnclosedLoopAnalyzer(request.image, is_base64=True)
        loop_results = loop_analyzer.analyze(debug=True)

        curvature_analyzer = CursiveCurvatureAnalyzer(request.image, is_base64=True)
        curvature_results = curvature_analyzer.analyze(debug=True)

        # Calligraphic Analysis
        stroke_width_analyzer = StrokeWidthAnalyzer(request.image, is_base64=True)
        stroke_width_results = stroke_width_analyzer.analyze(debug=True)

        flourish_analyzer = FlourishAnalyzer(request.image, is_base64=True)
        flourish_results = flourish_analyzer.analyze(debug=True)

        artistic_analyzer = CalligraphicAnalyzer(request.image, is_base64=True)
        artistic_results = artistic_analyzer.analyze(debug=True)

        # Shorthand Analysis
        continuity_analyzer = StrokeContinuityAnalyzer(request.image, is_base64=True)
        continuity_results = continuity_analyzer.analyze(debug=True)

        smooth_curves_analyzer = StrokeSmoothnessAnalyzer(request.image, is_base64=True)
        smooth_curves_results = smooth_curves_analyzer.analyze(debug=True)

        # Print Analysis
        vertical_alignment_analyzer = VerticalAlignmentAnalyzer(request.image, is_base64=True)
        vertical_alignment_results = vertical_alignment_analyzer.analyze(debug=True)

        letter_size_analyzer = LetterUniformityAnalyzer(request.image, is_base64=True)
        letter_size_results = letter_size_analyzer.analyze(debug=True)

        discrete_letter_analyzer = DiscreteLetterAnalyzer(request.image, is_base64=True)
        discrete_letter_results = discrete_letter_analyzer.analyze(debug=True)

        # - Score Calculation - #
        block_lettering_score = 0
        cursive_score = 0
        calligraphic_score = 0
        italic_score = 0
        shorthand_score = 0
        print_score = 0

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

        # Convert all results to Python native types
        response = {
            "processed_image": processed_image,
            # Block lettering
            "angularity": convert_numpy_types(angularity_results),
            "uppercase_ratio": convert_numpy_types(uppercase_results),
            "pen_pressure": convert_numpy_types(pen_pressure_results),

            # italic"
            "vertical_stroke_proportion": convert_numpy_types(vertical_stroke_results),
            "slant_angle": convert_numpy_types(slant_angle_results),
            "inter_letter_spacing": convert_numpy_types(spacing_results),

            # cursive"
            "stroke_connectivity": convert_numpy_types(connectivity_results),
            "enclosed_loop_ratio": convert_numpy_types(loop_results),
            "curvature_continuity": convert_numpy_types(curvature_results),

            # calligraphic"
            "stroke_width_variation": convert_numpy_types(stroke_width_results),
            "flourish_extension": convert_numpy_types(flourish_results),
            "artistic_consistency": convert_numpy_types(artistic_results),

            # shorthand
            "stroke_continuity": convert_numpy_types(continuity_results),
            "smooth_curves": convert_numpy_types(smooth_curves_results),

            # print
            "vertical_alignment": convert_numpy_types(vertical_alignment_results),
            "letter_size_uniformity": convert_numpy_types(letter_size_results),
            "discrete_letter": convert_numpy_types(discrete_letter_results),

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
