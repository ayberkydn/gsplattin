#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Usage: $0 <wandb-agent-id>" >&2
  exit 1
fi

agent_id="$1"

apptainer exec --nv \
	-B /arf/scratch/aaydin:/arf/home/aaydin \
	"../gsplattin.sif" \
	wandb agent "ayberkydn/models_sweep/${agent_id}"
