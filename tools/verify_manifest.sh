#!/usr/bin/env bash
# Checks tests/fixtures against tests/fixtures/MANIFEST.md: committed files by sha256, regenerated ones (*.safetensors, *.glb) by presence and size.
# Exit 0 when everything matches, 1 otherwise (each difference is printed).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fx="$root/tests/fixtures"
bad=0
while IFS='|' read -r _ f size sha committed _; do
  f="$(echo "$f" | tr -d ' `')"; size="$(echo "$size" | tr -d ' ')"; sha="$(echo "$sha" | tr -d ' `')"; committed="$(echo "$committed" | tr -d ' ')"
  [[ -n "$f" && "$f" != "file" && "$f" != -* ]] || continue
  if [[ ! -f "$fx/$f" ]]; then echo "missing: $f"; bad=1; continue; fi
  if [[ "$committed" == "yes" ]]; then
    [[ "$(sha256sum "$fx/$f" | cut -d' ' -f1)" == "$sha" ]] || { echo "changed: $f"; bad=1; }
  else
    [[ "$(stat -c %s "$fx/$f")" == "$size" ]] || { echo "other size: $f"; bad=1; }
  fi
done < <(grep '^| `' "$fx/MANIFEST.md")
[[ $bad == 0 ]] && echo "fixtures match MANIFEST.md"
exit $bad
