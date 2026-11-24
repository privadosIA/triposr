#!/usr/bin/env bash
# Starts the TripoSR HTTP service (src/TripoSR.Service), building it first when it is not built.
# Run it from anywhere: it finds the repository from its own location. See docs/http-service.md.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run-service.sh [options]

Starts triposr-service in the foreground (Ctrl+C or SIGTERM stops it). Every option has an environment variable;
an option wins over the variable.

  --model DIR        where the weights are (the layout of the container's /ckpt)       TRIPOSR_CKPT   (default: ~/models/model3d)
  --port N           port, on all interfaces                                           TRIPOSR_PORT   (default: 8790)
  --device D         cpu, cuda or cuda:N                                               TRIPOSR_DEVICE (default: cpu)
  --token-file FILE  file whose first line is the bearer token                         (or TRIPOSR_TOKEN; no token leaves the API open)
  --version V        the version string /health reports                                TRIPOSR_VERSION (default: 1)
  --build            build again even if the service is already built
  -h, --help         this text

A cuda device builds the service with the CUDA backend (-p:TorchBackend=cuda-linux), any other with the CPU one; if the
last build used the other backend, it is built again. The first build takes a few minutes.
Exit codes: 0 stopped normally, 2 wrong usage or a setting that cannot be used, anything else: the service's own.
USAGE
}

die() { echo "error: $*" >&2; exit 2; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${TRIPOSR_CKPT:-$HOME/models/model3d}"
port="${TRIPOSR_PORT:-8790}"
device="${TRIPOSR_DEVICE:-cpu}"
version="${TRIPOSR_VERSION:-1}"
token="${TRIPOSR_TOKEN:-}"
rebuild=0

need_value() { [[ $# -ge 2 ]] || die "option $1 needs a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       need_value "$@"; model="$2"; shift 2 ;;
    --port)        need_value "$@"; port="$2"; shift 2 ;;
    --device)      need_value "$@"; device="$2"; shift 2 ;;
    --version)     need_value "$@"; version="$2"; shift 2 ;;
    --token-file)  need_value "$@"; [[ -r "$2" ]] || die "--token-file: cannot read '$2'"
                   IFS= read -r token < "$2" || true; shift 2 ;;
    --build)       rebuild=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             echo "error: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
done

[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || die "the port must be a number from 1 to 65535, got '$port'"
[[ "$device" =~ ^(cpu|cuda|cuda:[0-9]+)$ ]] || die "the device must be cpu, cuda or cuda:N, got '$device'"
command -v dotnet >/dev/null || die "dotnet is not installed (the .NET 8 runtime or SDK is needed)"

# Build when needed, with the backend the device asks for.
backend=cpu; [[ "$device" == cuda* ]] && backend=cuda-linux
out="$root/src/TripoSR.Service/bin/Release/net8.0"
dll="${TRIPOSR_SERVICE_DLL:-$out/triposr-service.dll}"
marker="$out/.torch-backend"
if [[ -z "${TRIPOSR_SERVICE_DLL:-}" ]] && { (( rebuild )) || [[ ! -f "$dll" ]] || [[ "$(cat "$marker" 2>/dev/null)" != "$backend" ]]; }; then
  echo "building the service (backend: $backend) ..." >&2
  dotnet build "$root/src/TripoSR.Service" -c Release -p:TorchBackend="$backend" -v quiet -nologo >&2
  echo "$backend" > "$marker"
fi
[[ -f "$dll" ]] || die "the service is not built: $dll does not exist"

export TRIPOSR_CKPT="$model" TRIPOSR_PORT="$port" TRIPOSR_DEVICE="$device" TRIPOSR_VERSION="$version"
[[ -n "$token" ]] && export TRIPOSR_TOKEN="$token"

exec dotnet "$dll"
