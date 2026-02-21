#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${script_dir}/../gsplattin.sif"
BIND="/arf/scratch/aaydin:/arf/home/aaydin"
APPT="apptainer exec --nv -B ${BIND} ${SIF}"

$APPT python - <<'PY'
import torch
from gsplat import rasterization

d = torch.device("cuda")
n = 100
r = torch.randn
f = torch.nn.functional

out, _, _ = rasterization(
    means=r((n, 3), device=d),
    quats=f.normalize(r((n, 4), device=d), dim=-1),
    scales=torch.exp(r((n, 3), device=d)),
    opacities=torch.sigmoid(r((n,), device=d)),
    colors=r((n, 1, 3), device=d),
    viewmats=torch.eye(4, device=d)[None],
    Ks=torch.tensor([[[32, 0, 16], [0, 32, 16], [0, 0, 1]]], dtype=torch.float32, device=d),
    width=32,
    height=32,
    sh_degree=0,
    packed=False,
)
PY

$APPT python train.py
