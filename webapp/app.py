from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aiohttp import web

from webapp.evaluation import (
    evaluate_instance_segmentation,
    parse_yolo_segmentation_labels,
)
from webapp.tiling import (
    TiledGrainAnalyzer,
    encode_jpeg_data_url,
    summarize_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = Path(__file__).resolve().parent
LOCAL_LARGE_CHECKPOINT = (
    PROJECT_ROOT / "notebooks" / "model" / "large_v2" / "checkpoint_best_regular.pth"
)
SERVER_LARGE_CHECKPOINT = Path(
    "/home/multispector2/testdino/outputRFDetrV2/training/checkpoint_best_regular.pth"
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "MULTISPECTOR_CHECKPOINT",
        str(
            SERVER_LARGE_CHECKPOINT
            if SERVER_LARGE_CHECKPOINT.is_file()
            else LOCAL_LARGE_CHECKPOINT
        ),
    )
)
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_LABEL_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

LOGGER = logging.getLogger("multispector.web")


class AnalyzerProvider:
    def __init__(self, checkpoint: Path, optimize_fp16: bool) -> None:
        self.checkpoint = checkpoint
        self.optimize_fp16 = optimize_fp16
        self._analyzer: TiledGrainAnalyzer | None = None
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        if not self.checkpoint.is_file():
            return "missing-checkpoint"
        try:
            with self.checkpoint.open("rb") as checkpoint_file:
                header = checkpoint_file.read(4)
            if header.startswith(b"PK") and not zipfile.is_zipfile(self.checkpoint):
                return "corrupted-checkpoint"
        except OSError:
            return "unreadable-checkpoint"
        return "ready"

    def get(self) -> TiledGrainAnalyzer:
        if self._analyzer is None:
            with self._lock:
                if self._analyzer is None:
                    LOGGER.info("Loading RF-DETR checkpoint: %s", self.checkpoint)
                    self._analyzer = TiledGrainAnalyzer(
                        self.checkpoint,
                        optimize_fp16=self.optimize_fp16,
                    )
        return self._analyzer

    @property
    def loaded(self) -> bool:
        return self._analyzer is not None


def _float_field(fields: dict[str, str], name: str, default: float) -> float:
    try:
        return float(fields.get(name, default))
    except (TypeError, ValueError) as exc:
        raise web.HTTPBadRequest(text=f"Invalid value for {name}.") from exc


def _int_field(fields: dict[str, str], name: str, default: int) -> int:
    try:
        return int(fields.get(name, default))
    except (TypeError, ValueError) as exc:
        raise web.HTTPBadRequest(text=f"Invalid value for {name}.") from exc


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEBAPP_DIR / "templates" / "index.html")


async def health(request: web.Request) -> web.Response:
    provider: AnalyzerProvider = request.app["provider"]
    return web.json_response(
        {
            "status": provider.status,
            "model_loaded": provider.loaded,
            "model_variant": "RF-DETR Segmentation Large V2",
            "checkpoint": str(provider.checkpoint),
        }
    )


async def analyze(request: web.Request) -> web.Response:
    reader = await request.multipart()
    image_bytes: bytes | None = None
    label_bytes: bytes | None = None
    label_filename: str | None = None
    filename = "image"
    fields: dict[str, str] = {}

    async for part in reader:
        if part.name == "image":
            filename = part.filename or filename
            image_bytes = await part.read(decode=False)
            if len(image_bytes) > MAX_UPLOAD_BYTES:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=MAX_UPLOAD_BYTES,
                    actual_size=len(image_bytes),
                )
        elif part.name == "label":
            label_filename = part.filename
            label_bytes = await part.read(decode=False)
            if len(label_bytes) > MAX_LABEL_BYTES:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=MAX_LABEL_BYTES,
                    actual_size=len(label_bytes),
                )
        elif part.name:
            fields[part.name] = (await part.text()).strip()

    if not image_bytes:
        raise web.HTTPBadRequest(text="Please upload an image.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise web.HTTPBadRequest(text="The uploaded file is not a supported image.")
    if image_bgr.shape[0] * image_bgr.shape[1] > MAX_IMAGE_PIXELS:
        raise web.HTTPBadRequest(text="Image is too large; maximum is 40 megapixels.")

    threshold = min(max(_float_field(fields, "threshold", 0.35), 0.05), 0.95)
    tile_size = min(max(_int_field(fields, "tile_size", 640), 320), 1280)
    overlap = min(max(_float_field(fields, "overlap", 0.20), 0.10), 0.40)
    evaluation_iou = min(max(_float_field(fields, "evaluation_iou", 0.50), 0.10), 0.95)

    ground_truth = None
    if label_bytes is not None:
        if label_filename and not label_filename.lower().endswith(".txt"):
            raise web.HTTPBadRequest(text="Ground Truth label must be a YOLO .txt file.")
        try:
            label_text = label_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise web.HTTPBadRequest(text="Ground Truth label must be UTF-8 text.") from exc
        try:
            ground_truth = parse_yolo_segmentation_labels(
                label_text,
                image_width=int(image_bgr.shape[1]),
                image_height=int(image_bgr.shape[0]),
            )
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc

    provider: AnalyzerProvider = request.app["provider"]
    if provider.status != "ready":
        return web.json_response(
            {
                "error": (
                    f"Checkpoint is not usable ({provider.status}): "
                    f"{provider.checkpoint}. Download it again before analysis."
                )
            },
            status=500,
        )
    try:
        analyzer = await asyncio.to_thread(provider.get)
        result = await asyncio.to_thread(
            analyzer.analyze,
            image_bgr,
            threshold=threshold,
            tile_size=tile_size,
            overlap_ratio=overlap,
        )
    except Exception as exc:
        LOGGER.exception("Analysis failed")
        return web.json_response(
            {"error": f"Analysis failed: {exc}"},
            status=500,
        )

    summary = summarize_result(result)
    response: dict[str, Any] = {
        "filename": filename,
        "image_width": int(image_bgr.shape[1]),
        "image_height": int(image_bgr.shape[0]),
        **summary,
        "processing": {
            "elapsed_ms": round(result.elapsed_ms, 1),
            "tiles_processed": result.tiles_processed,
            "saturated_tiles": result.saturated_tiles,
            "raw_predictions": result.raw_predictions,
            "adaptive_parent_predictions_discarded": (
                result.raw_predictions - result.candidate_predictions
            ),
            "duplicates_removed": result.candidate_predictions - len(result.detections),
            "threshold": threshold,
            "tile_size": tile_size,
            "overlap": overlap,
            "device": analyzer.device,
            "fp16_optimized": analyzer.optimized,
            "model_variant": analyzer.model_variant,
        },
        "detections": [
            detection.as_dict(index)
            for index, detection in enumerate(result.detections, start=1)
        ],
        "annotated_image": encode_jpeg_data_url(result.annotated_image),
    }
    if ground_truth is not None:
        response["evaluation"] = evaluate_instance_segmentation(
            result.detections,
            ground_truth,
            iou_threshold=evaluation_iou,
        )
    return web.json_response(response)


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as exc:
        if request.path.startswith("/api/"):
            return web.json_response({"error": exc.text or exc.reason}, status=exc.status)
        raise


def create_app(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    optimize_fp16: bool = True,
) -> web.Application:
    app = web.Application(
        client_max_size=MAX_UPLOAD_BYTES + MAX_LABEL_BYTES,
        middlewares=[error_middleware],
    )
    app["provider"] = AnalyzerProvider(Path(checkpoint).resolve(), optimize_fp16)
    app.router.add_get("/", index)
    app.router.add_get("/api/health", health)
    app.router.add_post("/api/analyze", analyze)
    app.router.add_static("/static/", WEBAPP_DIR / "static", name="static")
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MULTISPECTOR grain inspection web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--no-fp16", action="store_true", help="Disable CUDA FP16 inference optimization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    web.run_app(
        create_app(args.checkpoint, optimize_fp16=not args.no_fp16),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
