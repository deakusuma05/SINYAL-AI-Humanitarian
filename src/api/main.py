"""
SINYAL Phase 3 — FastAPI Application Entrypoint (`src/api/main.py`).

Serves:
- GET  /               -> HTML5 Dashboard (`web/index.html`)
- GET  /api/health     -> Service & model artifact readiness
- GET  /api/metadata   -> Model metadata, frozen thresholds, safety principles
- POST /api/predict    -> Single-message real scikit-learn inference
- POST /api/batch      -> Batch inference (up to 50 messages)
- GET  /api/evaluation -> Frozen Phase 2 held-out test & validation metrics
- Static assets under `/css` and `/js`
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    EvaluationResponse,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
)
from src.api.services import (
    InvalidModelOutputError,
    ModelUnavailableError,
    check_health,
    get_frozen_evaluation,
    get_metadata,
    run_batch_prediction,
    run_single_prediction,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = PROJECT_ROOT / "web"
CSS_DIR = WEB_DIR / "css"
JS_DIR = WEB_DIR / "js"

app = FastAPI(
    title="SINYAL — AI-Powered Humanitarian Request Triage API",
    description=(
        "Research demonstration backend for SINYAL (IRDSC 2026 — Category D2, Team: Bismillah Bisa). "
        "Provides supervised multi-label NLP screening and uncertainty-aware human review routing. "
        "Not for emergency dispatch."
    ),
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Restrict CORS to local demonstration origins only (no wildcard '*' in production/demo config)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


@app.exception_handler(ModelUnavailableError)
async def handle_model_unavailable(
    request: Request, exc: ModelUnavailableError
) -> JSONResponse:
    """Return structured 503 error without exposing internal stack traces or local file paths."""
    return JSONResponse(
        status_code=503,
        content={
            "error": "model_unavailable",
            "message": str(exc),
            "application_mode": "research_demo",
        },
    )


@app.exception_handler(InvalidModelOutputError)
async def handle_invalid_model_output(
    request: Request, exc: InvalidModelOutputError
) -> JSONResponse:
    """Return structured 500 error when model output fails finite probability validation."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "invalid_model_output",
            "message": str(exc),
            "application_mode": "research_demo",
        },
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return clean 422 validation errors with human-readable details."""
    sanitized_errors = []
    for err in exc.errors():
        loc = " -> ".join(str(part) for part in err.get("loc", []))
        msg = str(err.get("msg", "Invalid input."))
        sanitized_errors.append({"field": loc, "message": msg})
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": sanitized_errors[0]["message"] if sanitized_errors else "Invalid request payload.",
            "details": sanitized_errors,
        },
    )


@app.get("/", include_in_schema=False)
async def serve_dashboard_index() -> FileResponse:
    """Serve the SINYAL HTML5 command-center interface."""
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "frontend_missing", "message": "Dashboard HTML file not found."},
        )
    return FileResponse(index_path, media_type="text/html")


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def api_health() -> HealthResponse:
    """Verify API and serialized scikit-learn model availability."""
    return check_health()


@app.get("/api/metadata", response_model=MetadataResponse, tags=["System"])
async def api_metadata() -> MetadataResponse:
    """Return frozen model metadata, validation-selected thresholds, and safety boundaries."""
    return get_metadata()


@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
async def api_predict(payload: PredictRequest) -> PredictResponse:
    """Run real multi-label inference and human-review policy evaluation on a single message."""
    return run_single_prediction(payload.message)


@app.post("/api/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def api_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    """Run real multi-label inference on a batch of up to 50 messages, preserving order."""
    return run_batch_prediction(payload.messages)


@app.get("/api/evaluation", response_model=EvaluationResponse, tags=["Evaluation"])
async def api_evaluation() -> EvaluationResponse:
    """Return frozen Phase 2 held-out test metrics and validation comparison summary."""
    try:
        return get_frozen_evaluation()
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": "evaluation_artifact_unavailable",
                "message": str(exc),
            },
        )


# Mount static directories if present
if CSS_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(CSS_DIR)), name="css")
else:
    CSS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/css", StaticFiles(directory=str(CSS_DIR)), name="css")

if JS_DIR.exists():
    app.mount("/js", StaticFiles(directory=str(JS_DIR)), name="js")
else:
    JS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/js", StaticFiles(directory=str(JS_DIR)), name="js")

