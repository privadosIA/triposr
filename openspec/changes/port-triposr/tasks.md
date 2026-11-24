# Tasks

## 1. Reference and weights

- [ ] 1.1 `tools/dump_reference.py` and `tools/golden_all.sh`: run the Python reference inside `mediaorquestator/model3d:1` over five pictures (skate, chest, lamp, trophy, cat) and dump the cut-out picture, the DINO tokens, the triplane tokens, the backbone output, the scene codes, the field on a fixed set of points, the density grid, the mesh and the final `.glb`; write `tests/fixtures/MANIFEST.md`.
- [ ] 1.2 Read `triposr/model.ckpt` without Python (move `PickleReader` and `TorchCheckpoint` from `~/Proyectos/lion-clap` to `musicgen-core`, or reference them) and write `model.safetensors` once in the golden run; a strict loader test that every key of the C# modules is in the file and the other way round.
- [ ] 1.3 `EngineInfo.RequiredFiles` and `/health` follow the real files (the weights, the DINO config, `u2net.onnx`).

## 2. The picture

- [ ] 2.1 Decoder for png, jpg and webp; the cut-out with ONNX Runtime and `u2net.onnx`; `resize_foreground(0.85)` and the composite over grey 0.5; parity with the reference's cut-out picture fixtures.

## 3. Image tokens

- [ ] 3.1 DINO ViT-B/16 (patch embedding, 12 blocks, interpolated position encoding for 512x512) with the names of `image_tokenizer.model.*`; parity of the tokens (max_abs <= 1e-4).

## 4. Triplane, backbone and scene codes

- [ ] 4.1 Triplane tokenizer (the learned embedding, 3 planes of 32x32 x 1024) and its `detokenize`.
- [ ] 4.2 `Transformer1D`: 16 blocks of self-attention, cross-attention to the image tokens and GEGLU feed-forward, with the checkpoint's names; parity of the backbone output.
- [ ] 4.3 `TriplaneUpsampleNetwork` (transposed convolution to 40 channels, 64x64); parity of the scene codes.

## 5. The field

- [ ] 5.1 `NeRFMLP` (120 -> 64 x 9, SiLU -> density and colour) and the triplane query (`grid_sample` on the three planes, concat, `exp` density with bias -1); parity on the fixed set of points.

## 6. The mesh

- [ ] 6.1 Density on the grid at 128, 192 and 256, threshold 25; marching cubes with the reference's order of axes and winding; colour per vertex queried at the vertices; parity of the triangles count and of the bounding box.
- [ ] 6.2 The finish of the container: upright (Y up), origin at the middle of the base, decimation to at most 500,000 triangles and the retreat to the next lower detail (`detail_used`).
- [ ] 6.3 `GlbWriter`: binary glTF with vertex colours, readable by `blender.ParseModel` of mediaOrquestator (the test opens the file the way the orchestrator does).

## 7. The service engine

- [ ] 7.1 `TripoSREngine : IModelEngine` replaces `NotPortedEngine` in `EngineFactory`; the model loads on the first job and stays on the card; one engine on the card at a time; `PortVersion` is set.
- [ ] 7.2 End to end: the five pictures through the service, the same details as the Python one; triangles, bounding box and Chamfer distance against the reference meshes; times and memory on the 5060 Ti measured and written in the README.
- [ ] 7.3 `bin/services/triposr` publish script (`dotnet publish`, cuda-linux) and the `ServiceSpec` in mediaOrquestator (task of that repository, listed in `docs/porting-plan.md`).
