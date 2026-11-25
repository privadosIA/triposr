#!/usr/bin/env bash
# Writes tests/fixtures/MANIFEST.md: the sha256 and the size of every fixture. tools/verify_manifest.sh checks it.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fx="$root/tests/fixtures"
out="$fx/MANIFEST.md"
{
  cat <<'HEAD'
# Fixtures of the golden run

Made by `tools/golden_all.sh` (the Python reference, TripoSR commit `107cefd`, inside `veritmo/model3d:1`, fp32 on the CPU) with
`tools/dump_reference.py`; the weights are not here and never are. Pictures are examples that come with TripoSR (MIT licence).

What is committed: the pictures, the cut-outs and inputs (`*.png`), the per-picture summaries (`*.json`), `ckpt_keys.json` and `versions.json`.
What is regenerated (too big for the repository, ignored by git): `*.safetensors` (stages, shared tensors, raw meshes) and `*.glb`.
The hash of a regenerated file can differ from this table between machines (the order of float sums changes with the threads and the
CPU): the tests load every fixture through `FixtureDirectory`, which checks its sha256 against this table and fails naming the file when they
differ, so after a deliberate golden run on another machine rewrite the table with `tools/write_manifest.sh`. `tools/verify_manifest.sh` checks
the committed files strictly and the regenerated ones only for being there, with the size.

| file | bytes | sha256 | committed |
|---|---:|---|:---:|
HEAD
  cd "$fx"
  find . -type f ! -name MANIFEST.md | LC_ALL=C sort | while read -r f; do
    f="${f#./}"
    case "$f" in *.safetensors|*.glb) c="no" ;; *) c="yes" ;; esac
    printf '| `%s` | %s | `%s` | %s |\n' "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)" "$c"
  done
} > "$out"
echo "wrote $out"
