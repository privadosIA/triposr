# tools/

The golden run: the only place Python exists in this repository, and only inside the reference container (`mediaorquestator/model3d:1`, offline, with the weights of `~/Proyectos/mediaOrquestator/models/model3d` mounted read-only).

Tasks 1.x of `openspec/changes/port-triposr` (all of these exist; `golden_all.sh` runs the golden run on five examples that come with TripoSR: chair, hamburger, horse, teapot (RGB: rembg cuts it out) and poly_fox (RGB), fp32 on the CPU, about 90 s):

| file | what |
|---|---|
| `dump_reference.py` | runs the reference over the pictures and writes, stage by stage, the tensors the port has to match (`.safetensors`) and the reference meshes |
| `golden_all.sh` | runs the container with that script and regenerates every fixture under `tests/fixtures/` (the weights converted to safetensors come with task 1.2, outside the repo) |
| `write_manifest.sh` / `verify_manifest.sh` | write and check `tests/fixtures/MANIFEST.md` (the sha256 of every fixture) |
| `gen_lewiner_tables.py` | one-off: writes `src/TripoSR.Net/Mesh/LewinerTables.cs` from scikit-image's look-up tables (read as data, never executed) |

Nothing here is needed to build, test or run the product once the fixtures exist.
