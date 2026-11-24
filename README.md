# TripoSR.Net

> Part of [mediaOrquestator](https://orquestator.xuxo.net/), where it makes the 3D models of the *models3d* category. **Status: scaffold.** The service answers the whole HTTP contract, but the engine is not ported yet: every job fails with *"TripoSR is not ported yet in this build"*. The plan is in [`openspec/changes/port-triposr`](openspec/changes/port-triposr) and in [`docs/porting-plan.md`](docs/porting-plan.md).

C# port (.NET 8 + [TorchSharp](https://github.com/dotnet/TorchSharp)) of [TripoSR](https://github.com/VAST-AI-Research/TripoSR), inference only: one picture in, one binary glTF (`.glb`) out. It replaces, as the product's engine, the Python container `docker/model3d` of mediaOrquestator (which stays as an optional fallback), the same way HeartMuLa, DiffRhythm2 and LAION-CLAP were migrated.

## What is in it

| part | where | what |
|---|---|---|
| Library | `src/TripoSR.Net/` | `EngineInfo` today (id, version, files of the weights, details); the model as it is ported (see the table below) |
| HTTP service | `src/TripoSR.Service/` | `GET /health`, `GET /engines`, `POST /generate`, `GET /jobs/{id}`, `GET /jobs/{id}/model`, `DELETE /jobs/{id}`, Swagger UI at `/docs`; **the same interface as `docker/model3d/server.py`**, so mediaOrquestator uses it with no change in how it asks. Default port 8790 |
| Tests | `tests/TripoSR.Net.Tests/` | xUnit; the service is tested in process (11 tests today) |
| Tools | `tools/` | the golden run (Python only inside the reference container, to make the fixtures once) |

## What it ports

| stage | in the reference (`tsr/`) | here (planned) |
|---|---|---|
| Picture | `ImagePreprocessor`: resize to 512, RGB; before it, `rembg` (u2net) cuts the object out and `resize_foreground(0.85)` composites it over grey 0.5 | `Picture/`: decode png/jpg/webp, cut-out with ONNX Runtime (`u2net.onnx`), the same resize and composite |
| Image tokens | `DINOSingleImageTokenizer`: DINO ViT-B/16 (`facebook/dino-vitb16`, 768 channels, interpolated position encoding) | `Vision/DinoViT.cs`: ViT-B/16 with the names of the checkpoint (`image_tokenizer.model.*`) |
| Triplane tokens | `Triplane1DTokenizer`: learned embedding, 3 planes of 32x32, 1024 channels | `Triplane/Tokenizer.cs` |
| Backbone | `Transformer1D`: 16 blocks, 16 heads of 64, cross-attention to the image tokens (768) | `Transformer/Transformer1D.cs` (self- and cross-attention blocks, GEGLU feed-forward) |
| Scene codes | `TriplaneUpsampleNetwork`: transposed convolution 1024 -> 40 channels, 32 -> 64 | `Triplane/UpsampleNetwork.cs` |
| Field | `NeRFMLP` (120 -> 64 x 9 hidden, SiLU -> density + colour) and `TriplaneNeRFRenderer.query_triplane` (`grid_sample` on the three planes, concat, `exp` density with bias -1, radius 0.87) | `Field/` |
| Surface | density on a grid (128, 192 or 256 per side), threshold 25, marching cubes; colour per vertex queried at the vertices | `Mesh/MarchingCubes.cs` (CPU first, the reference's own `skimage` order of axes and winding) |
| Finish | in the container: upright (Y up), origin at the middle of the base, no more than 500,000 triangles, colours kept | `Mesh/Grounding.cs`, `Mesh/Decimation.cs` |
| Output | `trimesh` `.glb` with vertex colours | `Mesh/GlbWriter.cs` |

`detail` is the resolution of the marching-cubes grid (it decides the triangles): marching-cubes grid 128, 192 and 256 (21,000 to 187,000 triangles on five objects).

## Build and run

```bash
dotnet build TripoSR.Net.sln -p:TorchBackend=cpu            # cpu | cuda-linux | cuda-windows
dotnet test  TripoSR.Net.sln -p:TorchBackend=cpu
scripts/run-service.sh --model ~/Proyectos/mediaOrquestator/models/model3d --port 8790
```

The weights are the ones of the container (`models/model3d` of mediaOrquestator: `triposr/config.yaml, triposr/model.ckpt, triposr/dino-vitb16-config.json, rembg/u2net.onnx`), never inside the repository. The shared model-agnostic code is in the sibling repository [`musicgen-core`](../musicgen-core) (`MusicGenCoreDir`).

## How it fits mediaOrquestator

The orchestrator makes a model with one HTTP call per picture; what it needs from a service is the contract in [`docs/http-service.md`](docs/http-service.md) (`engine_version` of a model is part of what decides it: this port says `triposr-net@<version of the port>`). The server starts and stops the service like the other .NET ones (published to `bin/services/`), or the container, as the person chooses.

The code of TripoSR and its weights are MIT; this port is Apache-2.0 (see `NOTICE`).
