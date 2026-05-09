"""SigLIP image/text embedder.

Endpoints
---------
GET  /health                      — liveness probe (after model load)
GET  /ready                       — readiness probe (kept distinct from
                                    /health so future drain logic can
                                    diverge the two semantics)
POST /embed/batch                 — main entry. Accepts a list of {id, url}
                                    items, returns L2-normalized embeddings.
POST /embed/image                 — single-image binary upload (raw body).
                                    For ad-hoc / debug calls.
POST /embed/text                  — text encoder side of SigLIP. Same vector
                                    space as image embeddings → enables
                                    "search images by text".

Auth
----
If SIGLIP_AUTH_TOKEN is set in the environment, all routes (except /health
and /ready) require an `Authorization: Bearer <token>` header. Strongly
recommended whenever the server is reachable from the public internet.

Vector contract
---------------
* dim is whatever the chosen SigLIP model emits (1152 for siglip2-so400m-patch14-384,
  1024 for siglip-large, 768 for siglip-base)
* values are L2-normalized → cosine similarity == dot product
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import secrets
import time
from typing import Annotated

import httpx
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Header, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOG = logging.getLogger("siglip")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

MODEL_PATH = os.environ.get("SIGLIP_MODEL_PATH") or os.environ.get(
    "SIGLIP_MODEL", "google/siglip2-so400m-patch14-384"
)
AUTH_TOKEN = os.environ.get("SIGLIP_AUTH_TOKEN", "").strip()
GPU_BATCH_SIZE = int(os.environ.get("SIGLIP_BATCH_SIZE", "32"))
FETCH_CONCURRENCY = int(os.environ.get("SIGLIP_FETCH_CONCURRENCY", "16"))
FETCH_TIMEOUT_S = float(os.environ.get("SIGLIP_FETCH_TIMEOUT_S", "20"))
MAX_ITEMS_PER_REQUEST = int(os.environ.get("SIGLIP_MAX_ITEMS", "256"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# Model load (sync at import → uvicorn waits → /health stays unreachable
# until the GPU is warm and the weights are resident).
# ---------------------------------------------------------------------------

LOG.info("loading model %s on %s (%s)", MODEL_PATH, DEVICE, DTYPE)
_t0 = time.time()
# use_fast=False is load-bearing for vector space stability. transformers 4.52
# flips the image-processor default to fast, which yields slightly different
# embeddings — silent drift relative to the existing Vectorize index. Keep
# explicit even though 4.50.x defaults to slow, so a future transformers bump
# doesn't quietly fork the vector space. Same lock in infra/siglip-modal.
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=DTYPE).to(DEVICE).eval()
EMBED_DIM = int(model.config.projection_dim) if hasattr(model.config, "projection_dim") else int(
    model.config.vision_config.hidden_size
)
LOG.info("model ready dim=%d in %.1fs", EMBED_DIM, time.time() - _t0)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class BatchItem(BaseModel):
    id: str = Field(..., description="caller-controlled id (e.g. an image's primary key)")
    url: str = Field(..., description="absolute http(s) URL the embedder fetches")


class BatchRequest(BaseModel):
    items: list[BatchItem]


class EmbedResult(BaseModel):
    id: str
    embedding: list[float]


class EmbedError(BaseModel):
    id: str
    error: str


class BatchResponse(BaseModel):
    model: str
    dim: int
    embeddings: list[EmbedResult]
    errors: list[EmbedError]


class TextRequest(BaseModel):
    texts: list[str]


class TextResponse(BaseModel):
    model: str
    dim: int
    embeddings: list[list[float]]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="siglip-embedder", version="1")


def _check_auth(authorization: str | None) -> None:
    if not AUTH_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if not secrets.compare_digest(authorization[len("Bearer "):], AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="invalid bearer token")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": MODEL_PATH, "dim": EMBED_DIM, "device": DEVICE}


@app.get("/ready")
def ready() -> dict:
    return {"ready": True}


# --- image batch ----------------------------------------------------------


async def _fetch_one(client: httpx.AsyncClient, item: BatchItem) -> tuple[str, Image.Image | str]:
    try:
        r = await client.get(item.url, timeout=FETCH_TIMEOUT_S, follow_redirects=True)
        if r.status_code != 200:
            return item.id, f"fetch_status_{r.status_code}"
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return item.id, img
    except (httpx.HTTPError, UnidentifiedImageError, OSError) as e:
        return item.id, f"fetch_failed:{type(e).__name__}"


@torch.inference_mode()
def _embed_images(images: list[Image.Image]) -> torch.Tensor:
    """Run SigLIP over a list of PIL images, return L2-normalized fp32 vectors."""
    out: list[torch.Tensor] = []
    for start in range(0, len(images), GPU_BATCH_SIZE):
        chunk = images[start : start + GPU_BATCH_SIZE]
        inputs = processor(images=chunk, return_tensors="pt").to(DEVICE)
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                feats = model.get_image_features(**inputs)
        else:
            feats = model.get_image_features(**inputs)
        feats = feats.float()
        feats = F.normalize(feats, p=2, dim=-1)
        out.append(feats.cpu())
    return torch.cat(out, dim=0) if out else torch.empty(0, EMBED_DIM)


@app.post("/embed/batch", response_model=BatchResponse)
async def embed_batch(
    req: BatchRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> BatchResponse:
    _check_auth(authorization)

    if not req.items:
        return BatchResponse(model=MODEL_PATH, dim=EMBED_DIM, embeddings=[], errors=[])
    if len(req.items) > MAX_ITEMS_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"too many items: {len(req.items)} > {MAX_ITEMS_PER_REQUEST}",
        )

    sem = asyncio.Semaphore(FETCH_CONCURRENCY)

    async def _bounded(client: httpx.AsyncClient, item: BatchItem):
        async with sem:
            return await _fetch_one(client, item)

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(_bounded(client, it) for it in req.items))

    ok_ids: list[str] = []
    ok_imgs: list[Image.Image] = []
    errors: list[EmbedError] = []
    for ident, payload in results:
        if isinstance(payload, str):
            errors.append(EmbedError(id=ident, error=payload))
        else:
            ok_ids.append(ident)
            ok_imgs.append(payload)

    embeds: list[EmbedResult] = []
    if ok_imgs:
        try:
            vecs = _embed_images(ok_imgs)
            for ident, vec in zip(ok_ids, vecs.tolist()):
                embeds.append(EmbedResult(id=ident, embedding=vec))
        except Exception as e:  # noqa: BLE001 — surface to caller, never crash worker
            LOG.exception("inference failed")
            for ident in ok_ids:
                errors.append(EmbedError(id=ident, error=f"inference_failed:{type(e).__name__}"))

    LOG.info(
        "embed_batch ok=%d err=%d total=%d", len(embeds), len(errors), len(req.items)
    )
    return BatchResponse(
        model=MODEL_PATH, dim=EMBED_DIM, embeddings=embeds, errors=errors
    )


# --- single image ---------------------------------------------------------


@app.post("/embed/image")
async def embed_image(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    _check_auth(authorization)

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")
    try:
        img = Image.open(io.BytesIO(body)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"decode_failed:{type(e).__name__}")

    vec = _embed_images([img])[0].tolist()
    return {"model": MODEL_PATH, "dim": EMBED_DIM, "embedding": vec}


# --- text ------------------------------------------------------------------


@torch.inference_mode()
def _embed_texts(texts: list[str]) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for start in range(0, len(texts), GPU_BATCH_SIZE):
        chunk = texts[start : start + GPU_BATCH_SIZE]
        inputs = processor(
            text=chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(DEVICE)
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                feats = model.get_text_features(**inputs)
        else:
            feats = model.get_text_features(**inputs)
        feats = feats.float()
        feats = F.normalize(feats, p=2, dim=-1)
        out.append(feats.cpu())
    return torch.cat(out, dim=0) if out else torch.empty(0, EMBED_DIM)


@app.post("/embed/text", response_model=TextResponse)
async def embed_text(
    req: TextRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> TextResponse:
    _check_auth(authorization)

    if not req.texts:
        return TextResponse(model=MODEL_PATH, dim=EMBED_DIM, embeddings=[])
    if len(req.texts) > MAX_ITEMS_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"too many texts: {len(req.texts)} > {MAX_ITEMS_PER_REQUEST}",
        )

    vecs = _embed_texts(req.texts).tolist()
    return TextResponse(model=MODEL_PATH, dim=EMBED_DIM, embeddings=vecs)
