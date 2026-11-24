# tools/

The golden run: the only place Python exists in this repository, and only inside the reference container (`mediaorquestator/model3d:1`, offline, with the weights of `~/Proyectos/mediaOrquestator/models/model3d` mounted read-only).

Planned (tasks 1.x of `openspec/changes/port-triposr`):

| file | what |
|---|---|
| `dump_reference.py` | runs the reference over the pictures and writes, stage by stage, the tensors the port has to match (`.safetensors`) and the reference meshes |
| `golden_all.sh` | runs the container with that script and regenerates every fixture under `tests/fixtures/`, and the weights converted to safetensors (outside the repo) |
| `write_manifest.sh` / `verify_manifest.sh` | write and check `tests/fixtures/MANIFEST.md` (the sha256 of every fixture) |

Nothing here is needed to build, test or run the product once the fixtures exist.
