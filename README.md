# TripoSR.Net

> Part of [mediaOrquestator](https://orquestator.xuxo.net/), where it makes the 3D models of the *models3d* category. **Status: ported.** Every stage of the model matches the Python reference and the five test pictures come out of the service as the container's models (see [Checked against the reference](#checked-against-the-reference)). What is left is on the mediaOrquestator side (the `ServiceSpec` and the publish entry; see [`docs/porting-plan.md`](docs/porting-plan.md)). The change is in [`openspec/changes/port-triposr`](openspec/changes/port-triposr).

C# port (.NET 8 + [TorchSharp](https://github.com/dotnet/TorchSharp)) of [TripoSR](https://github.com/VAST-AI-Research/TripoSR), inference only: one picture in, one binary glTF (`.glb`) out. It replaces, as the product's engine, the Python container `docker/model3d` of mediaOrquestator (which stays as an optional fallback), the same way HeartMuLa, DiffRhythm2 and LAION-CLAP were migrated.

## What is in it

| part | where | what |
|---|---|---|
| Library | `src/TripoSR.Net/` | `TripoSRPipeline` (picture in, finished model out), the network stage by stage (see the table below), `GlbWriter`, `EngineInfo` (id, version, files of the weights, details) |
| HTTP service | `src/TripoSR.Service/` | `GET /health`, `GET /engines`, `POST /generate`, `GET /jobs/{id}`, `GET /jobs/{id}/model`, `DELETE /jobs/{id}`, Swagger UI at `/docs`; **the same interface as `docker/model3d/server.py`**, so mediaOrquestator uses it with no change in how it asks. Default port 8790 |
| Tests | `tests/TripoSR.Net.Tests/` | xUnit: parity of every stage against the golden run, the service in process; the ones that need the weights are skipped, saying so, without `TRIPOSR_CKPT` |
| Tools | `tools/` | the golden run (Python only inside the reference container, to make the fixtures once) |

## What it ports

| stage | in the reference (`tsr/`) | here |
|---|---|---|
| Picture | `ImagePreprocessor`: resize to 512, RGB; before it, `rembg` (u2net) cuts the object out and `resize_foreground(0.85)` composites it over grey 0.5 | `Picture/`: SkiaSharp decoder, cut-out with ONNX Runtime (`u2net.onnx`; Pillow's Lanczos resize and paste reproduced), the same framing and composite |
| Image tokens | `DINOSingleImageTokenizer`: DINO ViT-B/16 (`facebook/dino-vitb16`, 768 channels, interpolated position encoding) | `Vision/DinoViT.cs`: ViT-B/16 with the names of the checkpoint (`image_tokenizer.model.*`) |
| Triplane tokens | `Triplane1DTokenizer`: learned embedding, 3 planes of 32x32, 1024 channels | `Triplane/TriplaneTokenizer.cs` |
| Backbone | `Transformer1D`: 16 blocks, 16 heads of 64, cross-attention to the image tokens (768) | `Transformer/Transformer1D.cs` (self- and cross-attention blocks, GEGLU feed-forward) |
| Scene codes | `TriplaneUpsampleNetwork`: transposed convolution 1024 -> 40 channels, 32 -> 64 | `Triplane/TriplaneUpsampleNetwork.cs` |
| Field | `NeRFMLP` (120 -> 64 x 9 hidden, SiLU -> density + colour) and `TriplaneNeRFRenderer.query_triplane` (`grid_sample` on the three planes, concat, `exp` density with bias -1, radius 0.87) | `Field/` (`NeRFMLP`, `TriplaneField`) |
| Surface | density on a grid (128, 192 or 256 per side), threshold 25, marching cubes; colour per vertex queried at the vertices | `Mesh/MarchingCubes.cs`: a port of scikit-image's Lewiner marching cubes, on the CPU, in the reference's order of axes and winding |
| Finish | in the container: upright (Y up), origin at the middle of the base, no more than 500,000 triangles, colours kept | `Mesh/ModelFinish.cs` (a model over the limit is made again at the next lower detail; the container never decimates) |
| Output | `trimesh` `.glb` with vertex colours | `Mesh/GlbWriter.cs` |

`detail` is the resolution of the marching-cubes grid (it decides the triangles): marching-cubes grid 128, 192 and 256 (21,000 to 187,000 triangles on five objects).

## Checked against the reference

The golden run (`tools/golden_all.sh`, the Python reference in `mediaorquestator/model3d:1`, fp32 on the CPU) over five pictures that come with TripoSR (chair, hamburger, horse, teapot, poly_fox: three with transparency, two that the cut-out model has to cut) leaves the fixtures of `tests/fixtures/`. On the CPU:

| stage | against the reference |
|---|---|
| cut-out (u2net on ONNX Runtime) | identical, pixel for pixel |
| 512x512 input, DINO tokens | max_abs 4e-7 and 3e-6 |
| triplane tokens, scene codes, field (density, colour) | identical |
| backbone | cos_sim 1.000000; max_abs 2e-3 on values up to 2,800, which is the reference's own noise (it differs from itself by 1.3e-3 between one thread and twenty) |
| surface (15 meshes: 5 pictures x 3 details) | same vertices, same faces, same colours |
| `.glb` | the binary chunk is byte for byte the container's |
| through the service (15 jobs) | same triangles, same bounding box, Chamfer distance 5e-8 of the size |

On the card (RTX 5060 Ti, CUDA) the float noise moves the surface by a hair: the triangles differ by at most 0.015 % and the Chamfer distance is 4e-6 of the size (the limit of the spec is 1e-3).

**Speed and memory** (RTX 5060 Ti, `TRIPOSR_DEVICE=cuda`, the five pictures at the three details, twice): once the model is loaded a model takes 0.45 to 0.73 s at `low`, 0.69 to 0.97 s at `normal` and 1.15 to 1.53 s at `high`; the first job also pays the load of the weights (3.1 s at `low`). The Python container takes 1 to 2 s. The peak memory of the card was about 6.0 GB above what it had before (8,550 MiB with 2,545 MiB used by other programs): more than the 3.4 GB the container needs, because the weights (1.6 GB) and the cached blocks of the allocator stay on the card between jobs. On the CPU a model takes some 5 to 17 s (measured inside the test suite, with other tests running, so only a rough figure).

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
