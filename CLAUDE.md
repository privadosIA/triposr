# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**Ported.** The whole model is in `src/TripoSR.Net` (picture, DINO, triplane, backbone, field, marching cubes, finish, `.glb`) and `src/TripoSR.Service` makes models with it (`TripoSREngine`, loaded on the first job). Every stage is tested against the fixtures of the golden run (`tests/fixtures/`, made by `tools/golden_all.sh`); the tests that need the weights read `TRIPOSR_CKPT` and are skipped without it. What is left of the OpenSpec change `port-triposr` (`openspec/changes/port-triposr`) is task 7.3, which lives in mediaOrquestator (the `ServiceSpec` and the entry in `scripts/get-services.sh`), and the archive. See `docs/porting-plan.md`.

## Build and test

```bash
dotnet build TripoSR.Net.sln -p:TorchBackend=cpu          # TorchBackend: cpu | cuda-linux | cuda-windows
dotnet test  TripoSR.Net.sln -p:TorchBackend=cpu
scripts/run-service.sh --model ~/Proyectos/mediaOrquestator/models/model3d --port 8790   # the HTTP service (docs/http-service.md)
dotnet tool restore && TorchBackend=cpu dotnet docfx docs/docfx.json --warningsAsErrors            # DocFX site into docs/_site
```

Layout (like `~/Proyectos/lion-clap` and `~/Proyectos/heartmula`): `src/TripoSR.Net` (library, namespace `TripoSR`), `src/TripoSR.Service` (ASP.NET Core service; `openapi.yaml` and the Swagger UI bundle embedded), `tests/TripoSR.Net.Tests` (xUnit, also tests the service in process), `tools/` (the golden run), `scripts/` (start script and systemd unit). The shared code lives in the sibling repository `~/Proyectos/musicgen-core` (`MusicGenCoreDir`, overridable with `-p:MusicGenCoreDir=...`); it is not part of the solution. Do not add `musicgen-core` projects to the `.sln` (`dotnet sln add` does it for referenced projects: remove them).

## Goal

A C# (.NET 8 + TorchSharp) inference-only port of TripoSR (https://github.com/VAST-AI-Research/TripoSR), with the same HTTP contract as the Python container `docker/model3d` of mediaOrquestator (`~/Proyectos/mediaOrquestator`), so the orchestrator can use it as an engine of the product with no Python. The golden reference is that container's image `mediaorquestator/model3d:1` (engine `triposr`, commit `107cefd`); its `docker/model3d/server.py` is the contract and the finishing of the model (upright, origin at the base, triangle limit, colours, GLB). The full rules (parity thresholds, scope, what is not ported) are in `openspec/config.yaml`.

## Workflow: OpenSpec (spec-driven)

Changes are planned and tracked with OpenSpec (`openspec/config.yaml`, `schema: spec-driven`). `openspec/specs/` holds the current specs, `openspec/changes/` the ones in flight and `openspec/changes/archive/` the completed ones. Commit right after every archive. Slash commands (`/opsx:explore`, `/opsx:propose`, `/opsx:update`, `/opsx:apply`, `/opsx:sync`, `/opsx:archive`) and the skills live in `.claude/`. Normal flow: explore → propose → apply → archive. The first change, `port-triposr`, is applied but for task 7.3.
