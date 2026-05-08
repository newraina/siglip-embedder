# siglip-embedder

A standalone HTTP server that turns images (and text) into [SigLIP](https://huggingface.co/google/siglip2-so400m-patch14-384) embeddings, packaged as a single self-contained Docker image. No volumes, init containers, or external storage required — the model weights are baked in, the container starts cleanly on any host with an NVIDIA GPU.

Designed for periodic offline batch jobs (gallery backfill, recommendation index rebuilds) but works equally well as a long-running service.

## What you get

- **`POST /embed/batch`** — fetch a list of `{id, url}` items in parallel, return L2-normalized embeddings ready to upsert into a vector DB
- **`POST /embed/image`** — single image (raw body) → vector
- **`POST /embed/text`** — text encoder side, same vector space → enables "search images by text" with the same index
- **`GET /health` / `GET /ready`** — k8s-friendly probes
- Optional bearer-token auth via `SIGLIP_AUTH_TOKEN`
- fp16 inference, configurable GPU batch size, concurrent URL fetches

Default model: `google/siglip2-so400m-patch14-384` → **1152-dim** vectors, ~3 GB VRAM, fits any modern card.

## Pre-built images

Public images on GHCR, built by GitHub Actions on every release tag:

```
ghcr.io/newraina/siglip-embedder:latest
ghcr.io/newraina/siglip-embedder:v1.0.0
ghcr.io/newraina/siglip-embedder:main
```

## Quick start

```bash
docker run --rm --gpus=all -p 8000:8000 \
  -e SIGLIP_AUTH_TOKEN=your-secret \
  ghcr.io/newraina/siglip-embedder:latest

# Wait ~15 s for model load, then:
curl http://localhost:8000/health

curl -X POST http://localhost:8000/embed/batch \
  -H "authorization: Bearer your-secret" \
  -H "content-type: application/json" \
  -d '{"items":[
    {"id":"a","url":"https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"},
    {"id":"b","url":"https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"}
  ]}'
```

Response shape:

```json
{
  "model": "google/siglip2-so400m-patch14-384",
  "dim": 1152,
  "embeddings": [
    {"id": "a", "embedding": [0.012, -0.043, ...]}
  ],
  "errors": [
    {"id": "b", "error": "fetch_status_404"}
  ]
}
```

## Configuration

All via environment variables. None are required.

| Env | Default | Purpose |
|---|---|---|
| `SIGLIP_MODEL_PATH` | (set in image) | Local path the image was baked with |
| `SIGLIP_MODEL` | `google/siglip2-so400m-patch14-384` | Hugging Face id (only honored if `SIGLIP_MODEL_PATH` is unset and you re-download at runtime) |
| `SIGLIP_AUTH_TOKEN` | unset | If set, `Authorization: Bearer <token>` is required on every route except `/health` and `/ready` |
| `SIGLIP_BATCH_SIZE` | `32` | GPU forward-pass batch. Raise on 24 GB+ cards for throughput |
| `SIGLIP_FETCH_CONCURRENCY` | `16` | Concurrent URL fetches per `/embed/batch` request |
| `SIGLIP_FETCH_TIMEOUT_S` | `20` | Per-URL fetch timeout |
| `SIGLIP_MAX_ITEMS` | `256` | Hard cap on items per request |
| `PORT` | `8000` | Bind port |

## Vector DB integration

Embeddings are L2-normalized at the server, so cosine similarity reduces to dot product on the storage side. Direct mappings:

- **Cloudflare Vectorize**: `--dimensions=1152 --metric=cosine`
- **pgvector**: `vector(1152)` with `vector_cosine_ops`
- **Pinecone / Qdrant / Milvus**: 1152, cosine

## Building

```bash
# Default model (SigLIP 2 so400m, 1152d)
docker build -t siglip-embedder .

# Different model (e.g. base, smaller / faster)
docker build --build-arg MODEL_ID=google/siglip-base-patch16-224 \
             -t siglip-embedder:base .

# Gated model — pass the HF token via build secret, then uncomment the
# matching line in the Dockerfile
DOCKER_BUILDKIT=1 docker build --secret id=HF_TOKEN,src=$HOME/.cache/huggingface/token \
             -t siglip-embedder .
```

## Deploying

The image is platform-agnostic. Examples:

- **Plain `docker run`**: see Quick start above
- **Kubernetes**: [`examples/k8s/deployment.yaml`](examples/k8s/deployment.yaml) (originally written for SuanLi, easily adaptable)
- **SuanLi pure-image deploy**: paste the image URL, port `8000`, GPU=1, set `SIGLIP_AUTH_TOKEN` env var
- **RunPod / Vast.ai / Lambda**: same — port 8000, GPU instance, env var for auth

## License

[MIT](LICENSE)
