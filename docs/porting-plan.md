# Porting plan

The model, stage by stage, with what each stage is in the reference and where it goes here. Every stage ends in a parity test against fixtures of the golden run (see `openspec/config.yaml` for the thresholds). The tasks, in order, are in `openspec/changes/port-triposr/tasks.md`.

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

## Checking it

1. **Golden run** (`tools/`, needs Docker and the weights, offline): the Python reference runs inside `mediaorquestator/model3d:1` over a handful of pictures (the five objects of the tests of mediaOrquestator: skate, chest, lamp, trophy, cat) and dumps, stage by stage, the tensors that the port has to match: the cut-out picture, the image tokens, the triplane tokens, the backbone output, the scene codes, the field on a fixed set of points, the density grid and the mesh. It also converts the weights to safetensors once, with the names of the checkpoint.
2. **Parity tests** in `tests/`: each stage loads its fixtures through `FixtureDirectory` (checked against `tests/fixtures/MANIFEST.md`), runs the C# stage and compares (fp32, max_abs <= 1e-4, cos_sim >= 0.9999).
3. **End to end**: the same five pictures through the service; the triangles, the bounding box and a Chamfer distance against the reference meshes.

## What stays Python

Only `tools/dump_reference.py`, inside the reference container, to make the fixtures. Nothing the product, its tests or its build need.
