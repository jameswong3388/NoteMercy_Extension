from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from helper import preprocess_image
from lib_py.block_lettering.angularity import BlockLetterAnalyzer

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


@app.post("/api/v1/extract")
async def analyze_image(request: ImageRequest):
    try:
        # Preprocess the image
        processed_image = preprocess_image(request.image)
        
        # Block Lettering
        analyzer = BlockLetterAnalyzer(request.image)
        angularity_results = analyzer.analyze(debug=True)

        return {
            "processed_image": processed_image,
            "angularity": angularity_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
