# Proposal

## Why

The 3D models of mediaOrquestator are made today by a Python container (`docker/model3d`, image `mediaorquestator/model3d:1`) that wraps TripoSR. The product is Go and .NET: HeartMuLa, DiffRhythm2 and LAION-CLAP were migrated from Python to .NET services (`bin/services/`) and their containers stay only as an optional path. The 3D engines are the next ones. This repository is the .NET port of TripoSR; the container remains as a fallback.

## What Changes

- A library `TripoSR.Net` (.NET 8 + TorchSharp) that ports the model, stage by stage, with parity tests against fixtures made once by the Python reference (TripoSR at commit `107cefd`, inside `mediaorquestator/model3d:1`).
- The service `TripoSR.Service` (already scaffolded with the whole HTTP contract of `docker/model3d/server.py` and an engine that is not ported yet) gets its real engine, so it answers every call with the same shapes and the same finish of the model (upright, origin at the base, at most 500,000 triangles, colours).
- Weights are read as they are in `models/model3d` (nothing is downloaded and nothing is copied into the repository); the Python is confined to `tools/`, inside the reference container.
- **Not ported:** Training, evaluation, the texture baking of the reference (`tsr/bake_texture.py`: the container never uses it), the CUDA `torchmcubes` extension (a CPU marching cubes gives the same mesh).

## Capabilities

### New Capabilities

- `triposr-service-api`: the HTTP contract and what the service does with a job.
- `triposr-weights`: the checkpoint is read without Python, strictly, and the weights folder is the container's.
- `triposr-model-parity`: each stage of the model matches the reference within the thresholds, and the mesh and the `.glb` match what the container makes.

## Impact

- This repository (`src/TripoSR.Net`, `src/TripoSR.Service`, `tests/`, `tools/`, `docs/`).
- `~/Proyectos/musicgen-core` (the shared loader and parity code; the pickle reader moves there from `lion-clap` if it is needed).
- mediaOrquestator, in its own change (not here): the `ServiceSpec` of the service in `bin/services`, a per-engine port when the two engines run as two services (the container has one port for both), the publish script in `scripts/get-services.sh`, and the choice between service and container in Environment.
