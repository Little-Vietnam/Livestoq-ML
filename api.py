"""
FastAPI server exposing the Livestock ML Pipeline.

Endpoints:
    GET  /health                 → pipeline readiness check
    POST /analyze                → dimension + weight from side image
    POST /analyze/teeth          → age prediction from teeth image
    POST /analyze/skin           → skin disease detection from side image
    POST /analyze/full           → combined analysis (side + optional teeth)

Run:
    python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import cv2
import numpy as np
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import PipelineConfig
from pipeline.main import LivestockPipeline
from models.age_predictor import AgePredictor
from models.skin_disease_detector import SkinDiseaseDetector

app = FastAPI(
    title="Livestoq ML API",
    description="Livestock dimension, weight, age, and skin disease analysis from images",
    version="2.0.0",
)

# Allow CORS from Next.js dev server and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise models once at startup
pipeline: Optional[LivestockPipeline] = None
age_predictor: Optional[AgePredictor] = None
skin_detector: Optional[SkinDiseaseDetector] = None


def _save_temp(contents: bytes, content_type: str = "") -> str:
    """Save bytes to a temp file and return path."""
    suffix = ".jpg" if "jpeg" in content_type else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        return tmp.name


def _jsonable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@app.on_event("startup")
def startup():
    global pipeline, age_predictor, skin_detector
    config = PipelineConfig()
    config.device = "cpu"
    config.debug = False
    pipeline = LivestockPipeline(config)
    age_predictor = AgePredictor()
    skin_detector = SkinDiseaseDetector()
    print("[API] All models initialized and ready.")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "age_predictor_ready": age_predictor is not None,
        "skin_detector_ready": skin_detector is not None,
    }


# ─── Original dimension + weight analysis (side image) ──────────────────

@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    breed: str = Form("generic"),
):
    """
    Analyse a livestock side image → dimensions + weight prediction.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        tmp_path = _save_temp(contents, image.content_type or "")

        try:
            results = pipeline.run(
                image_path=tmp_path,
                breed=breed,
                save_visualization=False,
            )

            return JSONResponse(content=_jsonable({
                "success": True,
                "image_size": results.get("image_size", {}),
                "breed": results.get("breed", breed),
                "segmentation": results.get("segmentation", {}),
                "distance": results.get("distance", {}),
                "keypoints": results.get("keypoints", {}),
                "pose": results.get("pose", {}),
                "dimensions": results.get("dimensions", {}),
                "dimension_confidence": results.get("dimension_confidence", {}),
                "weight": results.get("weight", {}),
            }))
        finally:
            os.unlink(tmp_path)

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[API Error] {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline processing failed: {str(e)}",
        )


# ─── Age prediction (teeth image) ───────────────────────────────────────

@app.post("/analyze/teeth")
async def analyze_teeth(
    image: UploadFile = File(...),
):
    """
    Predict cattle age from a teeth / mouth image.
    """
    if age_predictor is None:
        raise HTTPException(status_code=503, detail="Age predictor not initialized")

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        tmp_path = _save_temp(contents, image.content_type or "")

        try:
            result = age_predictor.predict(tmp_path)
            return JSONResponse(content=_jsonable({
                "success": True,
                "predicted_age_months": result.predicted_age_months,
                "age_range_months": list(result.age_range_months),
                "dentition_stage": result.dentition_stage,
                "wear_grade": result.wear_grade,
                "tooth_count": result.tooth_count,
                "confidence": result.confidence,
                "details": result.details,
            }))
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        print(f"[Age API Error] {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Age prediction failed: {str(e)}",
        )


# ─── Skin disease detection (side image) ────────────────────────────────

@app.post("/analyze/skin")
async def analyze_skin(
    image: UploadFile = File(...),
):
    """
    Detect skin diseases from a side-view image.
    """
    if skin_detector is None:
        raise HTTPException(status_code=503, detail="Skin detector not initialized")

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        tmp_path = _save_temp(contents, image.content_type or "")

        try:
            result = skin_detector.detect(tmp_path)
            return JSONResponse(content=_jsonable({
                "success": True,
                "overall_status": result.overall_status,
                "overall_confidence": result.overall_confidence,
                "skin_quality_score": result.skin_quality_score,
                "conditions": [
                    {
                        "name": c.name,
                        "confidence": c.confidence,
                        "severity": c.severity,
                        "affected_area_pct": c.affected_area_pct,
                        "description": c.description,
                    }
                    for c in result.conditions
                ],
                "details": result.details,
            }))
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        print(f"[Skin API Error] {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Skin disease detection failed: {str(e)}",
        )


# ─── Combined full analysis ─────────────────────────────────────────────

@app.post("/analyze/full")
async def analyze_full(
    side_image: UploadFile = File(...),
    teeth_image: Optional[UploadFile] = File(None),
    breed: str = Form("generic"),
):
    """
    Run all available analyses:
      - side_image (required): dimension + weight + skin disease
      - teeth_image (optional): age prediction

    Returns a combined result with all pipeline outputs.
    """
    if pipeline is None or skin_detector is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    response: dict = {
        "success": True,
        "analyses_run": [],
    }

    # ── Side image: dimension + weight + skin ────────────────────────
    try:
        side_contents = await side_image.read()
        if not side_contents:
            raise HTTPException(status_code=400, detail="Side image is empty")

        side_path = _save_temp(side_contents, side_image.content_type or "")

        try:
            # Dimension + weight pipeline
            dim_results = pipeline.run(
                image_path=side_path,
                breed=breed,
                save_visualization=False,
            )
            response["dimension_weight"] = {
                "image_size": dim_results.get("image_size", {}),
                "breed": dim_results.get("breed", breed),
                "segmentation": dim_results.get("segmentation", {}),
                "distance": dim_results.get("distance", {}),
                "keypoints": dim_results.get("keypoints", {}),
                "pose": dim_results.get("pose", {}),
                "dimensions": dim_results.get("dimensions", {}),
                "dimension_confidence": dim_results.get("dimension_confidence", {}),
                "weight": dim_results.get("weight", {}),
            }
            response["analyses_run"].append("dimension_weight")

            # Skin disease (same side image)
            skin_result = skin_detector.detect(side_path)
            response["skin_disease"] = {
                "overall_status": skin_result.overall_status,
                "overall_confidence": skin_result.overall_confidence,
                "skin_quality_score": skin_result.skin_quality_score,
                "conditions": [
                    {
                        "name": c.name,
                        "confidence": c.confidence,
                        "severity": c.severity,
                        "affected_area_pct": c.affected_area_pct,
                        "description": c.description,
                    }
                    for c in skin_result.conditions
                ],
                "details": skin_result.details,
            }
            response["analyses_run"].append("skin_disease")

        finally:
            os.unlink(side_path)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Full/Side Error] {type(e).__name__}: {e}")
        response["dimension_weight_error"] = str(e)

    # ── Teeth image: age prediction (optional) ───────────────────────
    if teeth_image is not None:
        try:
            teeth_contents = await teeth_image.read()
            if teeth_contents and len(teeth_contents) > 0:
                teeth_path = _save_temp(teeth_contents, teeth_image.content_type or "")
                try:
                    age_result = age_predictor.predict(teeth_path)
                    response["age_prediction"] = {
                        "predicted_age_months": age_result.predicted_age_months,
                        "age_range_months": list(age_result.age_range_months),
                        "dentition_stage": age_result.dentition_stage,
                        "wear_grade": age_result.wear_grade,
                        "tooth_count": age_result.tooth_count,
                        "confidence": age_result.confidence,
                        "details": age_result.details,
                    }
                    response["analyses_run"].append("age_prediction")
                finally:
                    os.unlink(teeth_path)
        except Exception as e:
            print(f"[Full/Teeth Error] {type(e).__name__}: {e}")
            response["age_prediction_error"] = str(e)

    return JSONResponse(content=_jsonable(response))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
