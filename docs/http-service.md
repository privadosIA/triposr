# The HTTP service

`triposr-service` (project `src/TripoSR.Service`) makes a 3D model from one picture with TripoSR. It is the same interface as the Python service `docker/model3d/server.py` of Veritmo, so the orchestrator uses it with no change in how it asks, and any other program can use it with `curl`. Open `http://HOST:PORT/docs` for the interactive page (Swagger UI, bundled, works offline); the machine-readable description is `http://HOST:PORT/openapi.yaml`.

It speaks plain HTTP and is meant for localhost or a trusted network. Put a reverse proxy in front of it if it must be reached from elsewhere, and set a token.

> The model is loaded when the first job asks for it and stays in memory (on the card with a cuda device); the first job pays for the load.

## Start it

```bash
scripts/run-service.sh --model ~/Proyectos/veritmo/models/model3d --port 8790
```

| variable | default | meaning |
|---|---|---|
| `TRIPOSR_CKPT` | `/ckpt` | where the weights are: the layout of the container's `/ckpt` (`triposr/config.yaml, triposr/model.ckpt, triposr/dino-vitb16-config.json, rembg/u2net.onnx`) |
| `TRIPOSR_TOKEN` | empty | bearer token; empty leaves the API open |
| `TRIPOSR_PORT` | `8790` | port |
| `TRIPOSR_HOST` | `0.0.0.0` | the address to listen on: `0.0.0.0` (all interfaces, as the Python container does) or `127.0.0.1` / `localhost` for this machine only. With no token, listen on loopback |
| `TRIPOSR_VERSION` | `1` | the version string `/health` reports (the version of a *model* is `triposr-net@<EngineInfo.PortVersion>`) |
| `TRIPOSR_DEVICE` | `cpu` | `cpu`, `cuda` or `cuda:N` (a CUDA device needs a CUDA build) |

## The calls

| call | answer |
|---|---|
| `GET /health` | `{ready, version, busy, queued, gpu, missing, engines}`: `ready` is false while a file of the weights is missing (named in `missing`) or the device does not work |
| `GET /engines` | `{engines: {triposr: {built, missing, ready, version}}}` |
| `POST /generate?engine=triposr&detail=low\|normal\|high` | the picture (png, jpg or webp, up to 20 MB) as the body -> `202 {id}`; `400` for another engine, a detail that is none, a body that is not a picture or weights that are not complete; `413` over 20 MB |
| `GET /jobs/{id}` | `{id, state (queued, running, done, failed), progress, error, seconds, engine, engine_version, detail, detail_used, resolution, triangles}` |
| `GET /jobs/{id}/model` | the `.glb` (`model/gltf-binary`); `409` until the job is done |
| `DELETE /jobs/{id}` | forgets the job and its model |

`detail` is the resolution of the marching-cubes grid (it decides the triangles). A model over 500,000 triangles is made again at the next lower detail and `detail_used` says which one it was made at. Every model leaves upright (Y up) with its origin at the middle of its base and its colours.

## How Veritmo uses it

The server runs it as a service of `bin/services/` (published with `dotnet publish`, like HeartMuLa and DiffRhythm2) or, with the Python container, the other way; both answer this contract. The `engine_version` of a model is part of what the orchestrator caches by: it never reuses a model made by another version of the port.
