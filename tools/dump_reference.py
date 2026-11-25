#!/usr/bin/env python3
"""The golden run: the Python reference (TripoSR at commit 107cefd, inside veritmo/model3d:1) over some pictures, dumping what
each stage of the C# port has to match. It is a migration tool: nothing the product, its tests or its build need once the fixtures exist.

Run it through tools/golden_all.sh (it mounts what this needs into the container). It uses the container's own `server.py` for the cut-out,
the finish of the model (upright, grounded, triangle limit) and the `.glb`, so the fixtures are what the container makes.

Per picture NAME it writes in OUT:
  NAME.cutout.png            the picture with the background cut out (rembg/u2net), RGBA, as server.cutout gives it
  NAME.input.png             what the model sees: resize_foreground(0.85) composited over grey 0.5, RGB, before the resize to 512
  NAME.stages.safetensors    rgb_cond (1,512,512,3), image_tokens (1,1025,768), backbone (1,1024,3072),
                             scene_codes (1,3,40,64,64), field_density, field_density_act, field_color (the fixed points of shared.*),
                             density_128 (128,128,128: the grid that marching cubes reads at detail "low")
  NAME.R.mesh.safetensors    the mesh as extract_mesh gives it at resolution R (128, 192, 256): vertices, faces, colors (float 0..1)
  NAME.D.glb                 the model the container makes for the detail D asked (low, normal, high), after the finish and the fallback
  NAME.json                  per detail: triangles and detail_used of that model, and the triangles of each raw mesh
and once: shared.safetensors (the fixed points, the triplane tokens), ckpt_keys.json (name, shape, dtype of every key of the checkpoint),
and versions.json (what ran).

The device is the CPU unless --device says otherwise: fp32 on the CPU is what the parity tests of the port compare to (the card's
convolutions use TF32 by default and would put a 1e-3 noise into the fixtures).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from safetensors.torch import save_file


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def tensors(**kw):
    import torch

    return {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).detach().cpu().contiguous() for k, v in kw.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pictures", required=True, help="a folder with NAME.png / NAME.jpg / NAME.jpeg")
    p.add_argument("--names", required=True, help="comma separated list of the pictures to run")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--points", type=int, default=4096, help="size of the fixed set of points for the field")
    args = p.parse_args()

    sys.path.insert(0, os.environ.get("TRIPOSR_DIR", "/opt/TripoSR"))
    sys.path.insert(0, os.environ.get("MODEL3D_SERVER_DIR", "/srv/model3d"))
    ckpt = os.environ.get("MODEL3D_CKPT", "/ckpt")
    os.environ.setdefault("TRIPOSR_DINO_CONFIG", os.path.join(ckpt, "triposr", "dino-vitb16-config.json"))
    os.makedirs(args.out, exist_ok=True)

    import torch
    from einops import rearrange

    import server  # the container's own: the cut-out, the finish of the model and the .glb
    from tsr.system import TSR
    from tsr.utils import resize_foreground, scale_tensor

    torch.set_num_threads(os.cpu_count() or 1)
    dev = args.device

    # --- the weights: the keys of the checkpoint (what the C# modules have to be named) ---
    weights = os.path.join(ckpt, "triposr", "model.ckpt")
    sd = torch.load(weights, map_location="cpu")
    # the sha256 of the raw bytes of each tensor is what the C# reader of the pickle is checked against (bit for bit, no 1.6 GB fixture)
    keys = [
        {"name": k, "shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", ""), "sha256": hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()}
        for k, v in sd.items()
    ]
    with open(os.path.join(args.out, "ckpt_keys.json"), "w") as f:
        json.dump(keys, f, indent=1)
    log(f"checkpoint: {len(keys)} keys")
    del sd

    model = TSR.from_pretrained(os.path.join(ckpt, "triposr"), config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)  # as the container does
    model.to(dev)
    model.eval()

    # --- what does not depend on the picture ---
    rng = np.random.RandomState(1234)
    radius = model.renderer.cfg.radius
    points = torch.from_numpy(rng.uniform(-radius, radius, size=(args.points, 3)).astype(np.float32))
    with torch.no_grad():
        planes_embedding = model.tokenizer(1)  # (1, 1024, 3072)
    shared = tensors(points=points, triplane_tokens=planes_embedding)
    save_file(shared, os.path.join(args.out, "shared.safetensors"))

    versions = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": dev,
        "triposr_commit": os.environ.get("TRIPOSR_COMMIT", ""),
        "radius": radius,
        "threshold": 25.0,
        "points_seed": 1234,
        "details": server.DETAIL,
        "max_triangles": server.MAX_TRIANGLES,
    }
    try:
        import skimage
        import trimesh

        versions["skimage"], versions["trimesh"] = skimage.__version__, trimesh.__version__
    except ImportError:
        pass
    with open(os.path.join(args.out, "versions.json"), "w") as f:
        json.dump(versions, f, indent=1)

    names = [n for n in args.names.split(",") if n]
    for name in names:
        path = next((os.path.join(args.pictures, name + e) for e in (".png", ".jpg", ".jpeg", ".webp") if os.path.exists(os.path.join(args.pictures, name + e))), None)
        if path is None:
            raise SystemExit(f"no picture {name} in {args.pictures}")
        t0 = time.time()
        with open(path, "rb") as f:
            data = f.read()
        picture_sha = hashlib.sha256(data).hexdigest()

        # the picture, as server.run_job and TripoSR.generate do it
        rgba = server.cutout(data)
        rgba.save(os.path.join(args.out, f"{name}.cutout.png"))
        img = resize_foreground(rgba, 0.85)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img.save(os.path.join(args.out, f"{name}.input.png"))

        # TSR.forward, stage by stage (the same calls, in the same order)
        with torch.no_grad():
            rgb_cond = model.image_processor([img], model.cfg.cond_image_size)[:, None].to(dev)  # (1,1,512,512,3)
            tok = model.image_tokenizer(rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1))  # (1,1,768,1025)
            image_tokens = rearrange(tok, "B Nv C Nt -> B (Nv Nt) C", Nv=1)  # (1,1025,768)
            tokens = model.tokenizer(1)
            backbone = model.backbone(tokens, encoder_hidden_states=image_tokens)  # (1,1024,3072)
            planes = model.tokenizer.detokenize(backbone)  # (1,3,1024,32,32)
            scene_codes = model.post_processor(planes)  # (1,3,40,64,64)
            field = model.renderer.query_triplane(model.decoder, points.to(dev), scene_codes[0])
            # the field of the fixed points: raw density (before bias and exp), density_act, color
            density_128 = model.renderer.query_triplane(
                model.decoder,
                scale_tensor(
                    _grid(model, 128).to(dev),
                    (0, 1),
                    (-radius, radius),
                ),
                scene_codes[0],
            )["density_act"].reshape(128, 128, 128)
        stages = tensors(
            rgb_cond=rgb_cond[:, 0],
            image_tokens=image_tokens,
            backbone=backbone,
            scene_codes=scene_codes,
            field_density=field["density"],
            field_density_act=field["density_act"],
            field_color=field["color"],
            density_128=density_128,
        )
        save_file(stages, os.path.join(args.out, f"{name}.stages.safetensors"))
        log(f"{name}: stages done in {time.time() - t0:.1f}s")

        # the meshes at every resolution as extract_mesh gives them, and the model of each detail as the container leaves it
        meta = {"picture": os.path.basename(path), "picture_sha256": picture_sha, "raw": {}, "models": {}}
        raw_meshes = {}
        for d, res in server.DETAIL.items():
            t1 = time.time()
            with torch.no_grad():
                mesh = model.extract_mesh(scene_codes, True, resolution=res)[0]
            raw_meshes[d] = mesh
            save_file(
                tensors(vertices=np.asarray(mesh.vertices, dtype=np.float32), faces=np.asarray(mesh.faces, dtype=np.int64), colors=np.asarray(mesh.visual.vertex_colors, dtype=np.uint8).astype(np.float32) / 255.0),
                os.path.join(args.out, f"{name}.{res}.mesh.safetensors"),
            )
            meta["raw"][str(res)] = {"vertices": int(len(mesh.vertices)), "triangles": int(len(mesh.faces)), "seconds": round(time.time() - t1, 1)}
            log(f"{name}: mesh {res}: {len(mesh.faces)} triangles in {time.time() - t1:.1f}s")

        for asked in server.DETAIL:
            used, final = None, None
            for d in server.details_from(asked):
                m = raw_meshes[d].copy()
                m = server.upright_and_grounded(m)
                try:
                    server.check_mesh(m)
                    used, final = d, m
                    break
                except server.TooMany:
                    continue
                except server.Refused as e:
                    meta["models"][asked] = {"error": str(e)}
                    break
            if final is None:
                meta["models"].setdefault(asked, {"error": "every detail has too many triangles"})
                continue
            glb = server.export_glb(final)
            with open(os.path.join(args.out, f"{name}.{asked}.glb"), "wb") as f:
                f.write(glb)
            lo, hi = final.bounds
            meta["models"][asked] = {"detail_used": used, "triangles": int(len(final.faces)), "bounds_min": lo.tolist(), "bounds_max": hi.tolist(), "glb_bytes": len(glb)}
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        log(f"{name}: done in {time.time() - t0:.1f}s")


def _grid(model, resolution):
    """The grid of points of the isosurface helper (the same one extract_mesh builds), in 0..1."""
    model.set_marching_cubes_resolution(resolution)
    return model.isosurface_helper.grid_vertices


if __name__ == "__main__":
    main()
