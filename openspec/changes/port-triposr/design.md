# Design

## Context

What exists: the Python container (`docker/model3d/server.py`, `fetch.py`, `triposr-local.patch`) and its measured behaviour (see the README of mediaOrquestator's `docs/MODEL_GENERATION.md`): one model in 1 to 2 s with the engine loaded (marching-cubes grid 128, 192 and 256 (21,000 to 187,000 triangles on five objects)), 3.4 GB of VRAM. What exists here: the library with `EngineInfo`, the service with the full contract and `NotPortedEngine`, and 11 tests.

## Decisions

**1. The contract is frozen at the container's.** The service speaks exactly the calls, bodies, status codes and field names of `server.py`; `openapi.yaml` and a test that compares it with the route table keep it honest. The orchestrator does not change how it asks; it only learns where the service is.

**2. The port version decides the cache.** The orchestrator keys a model by its `engine_version`; this port says `triposr-net@<PortVersion>`, so a model of the container and a model of the port are never taken for each other, and `PortVersion` changes whenever a model would.

**3. Parity first, speed later.** Each stage is ported in fp32 on the CPU first and checked against fixtures (`max_abs <= 1e-4`, `cos_sim >= 0.9999`; the limit on `max_abs` is relative to the stage's largest magnitude when that is above 1, because fp32 noise grows with it: the backbone output reaches 2,800 and the Python reference differs from itself by 1.3e-3 between one thread and twenty); the CUDA path comes with an opt-in test, as in `lion-clap`. The mesh is checked by triangles count, bounding box and a Chamfer distance (marching cubes is exact on the same density; it is the density that carries the tolerance).

**4. Weights are the container's.** The files and their names are those of `models/model3d`; the C# modules replicate the keys of the checkpoint so the strict loader needs no mapping. The PyTorch pickle is read directly (the reader of `lion-clap`) and converted once to safetensors in the golden run.

**5. Python only in `tools/`.** The golden run needs the reference; nothing else does. Fixtures are in the repository (small) or regenerated (big), checked against a sha256 table; the weights never are.

## Risks / Trade-offs

- **Parity of a transformer this size in fp32** can drift by layers; the stages are checked one by one so a drift is found where it starts.
- **Memory and time on the card.** The reference needs 3.4 GB of VRAM; the port must not need more, and its first load time is measured, not assumed.

## Migration Plan

Nothing changes in mediaOrquestator until the port passes its end-to-end check; then its own change adds the service next to the container, and the person chooses (the container stays as the optional path). Rollback is not using the service.

## Open Questions

- Whether `marching cubes` on the CPU is fast enough at 256 per side (the container did it on the CPU too), or it goes to the card.
- One service per engine (this repository's port 8790) while the container has both engines behind one port: the orchestrator's `ServiceSpec` per engine is a change of mediaOrquestator.
