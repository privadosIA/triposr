#!/usr/bin/env bash
# The golden run: regenerates the fixtures under tests/fixtures/ with the Python reference inside veritmo/model3d:1.
# Needs Docker, the image and the weights folder; works offline. Nothing the product, its tests or its build need once the fixtures exist.
#
#   tools/golden_all.sh [--model DIR] [--orchestrator DIR] [--device cpu|cuda] [--names a,b,c] [--image IMG]
#
# --model         the weights folder (the container's /ckpt)         default ~/Proyectos/veritmo/models/model3d
# --orchestrator  the Veritmo checkout (its docker/model3d/server.py is the contract the fixtures follow)
# --names         the pictures (files of tests/fixtures/pictures)    default chair,hamburger,horse,teapot,poly_fox
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="$HOME/Proyectos/veritmo/models/model3d"
orch="$HOME/Proyectos/veritmo"
device="cpu"
names="chair,hamburger,horse,teapot,poly_fox"
image="veritmo/model3d:1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --orchestrator) orch="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --names) names="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown option $1" >&2; exit 2 ;;
  esac
done

[[ -f "$model/triposr/model.ckpt" ]] || { echo "error: no triposr/model.ckpt in $model" >&2; exit 2; }
[[ -f "$orch/docker/model3d/server.py" ]] || { echo "error: no docker/model3d/server.py in $orch" >&2; exit 2; }
docker image inspect "$image" >/dev/null 2>&1 || { echo "error: the image $image is not here" >&2; exit 2; }

fixtures="$root/tests/fixtures"
mkdir -p "$fixtures/pictures"

# The pictures are examples that come with TripoSR (MIT): taken from the reference image the first time, then kept.
IFS=, read -ra list <<<"$names"
missing=0
for n in "${list[@]}"; do
  ls "$fixtures/pictures/$n".* >/dev/null 2>&1 || missing=1
done
if [[ $missing == 1 ]]; then
  docker run --rm --user "$(id -u):$(id -g)" --entrypoint sh -v "$fixtures/pictures:/out" "$image" -c 'cp /opt/TripoSR/examples/* /out/'
  # keep only the ones the run uses
  for f in "$fixtures"/pictures/*; do
    b="$(basename "$f")"; keep=0
    for n in "${list[@]}"; do [[ "${b%.*}" == "$n" ]] && keep=1; done
    [[ $keep == 1 ]] || rm -f "$f"
  done
fi

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e USER=golden -e LOGNAME=golden -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -e NUMBA_CACHE_DIR=/tmp -e U2NET_HOME=/ckpt/rembg -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$model:/ckpt:ro" \
  -v "$orch/docker/model3d:/srv/model3d:ro" \
  -v "$root/tools:/tools:ro" \
  -v "$fixtures/pictures:/pictures:ro" \
  -v "$fixtures:/out" \
  --entrypoint python3 "$image" -I /tools/dump_reference.py \
  --pictures /pictures --names "$names" --out /out --device "$device"

# the container writes as the user but with a closed mode
chmod -R a+rX,u+w "$fixtures"
"$root/tools/write_manifest.sh"
