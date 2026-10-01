"""FastAPI app factory for verqen-labeler: /label, /label_dataset, /healthz."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from verqen_labeler import Labeler, __version__
from verqen_labeler.api.schemas import (
    HealthResponse,
    LabelDatasetRequest,
    LabelDatasetResponse,
    LabelRequest,
    LabelResponse,
    LabelingReportModel,
)

__all__ = ["create_api"]


def create_api(
    labeler: Labeler | None = None,
    *,
    categories: list[str] | None = None,
) -> FastAPI:
    """Build the FastAPI app for a labeler.

    Inject a ready ``labeler`` (handy in tests), or pass ``categories`` and
    the labeler is constructed from ``VERQEN_CHECKPOINT``. Raises
    ``ValueError`` when neither is given::

        import uvicorn
        from verqen_labeler import Labeler
        from verqen_labeler.api import create_api

        app = create_api(labeler=Labeler(categories=["politics", "sports"]))
        uvicorn.run(app, host="127.0.0.1", port=8000)
    """
    if labeler is None:
        if not categories:
            raise ValueError("create_api needs either labeler=... or categories=[...]")
        labeler = Labeler(categories)

    app = FastAPI(title="verqen-labeler", version=__version__)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok", categories=list(labeler.categories))

    @app.post("/label", response_model=LabelResponse)
    def label(request: LabelRequest) -> dict:
        try:
            result = labeler.label(request.text, request.context)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(result)

    @app.post("/label_dataset", response_model=LabelDatasetResponse)
    def label_dataset(request: LabelDatasetRequest) -> dict:
        try:
            results, report = labeler.label_dataset(request.texts, request.context)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"results": [asdict(result) for result in results], "report": asdict(report)}

    return app
