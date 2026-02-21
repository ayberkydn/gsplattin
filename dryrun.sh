#!/usr/bin/env bash
set -euo pipefail

SIF="../gsplattin.sif"
BIND="/arf/scratch/aaydin:/arf/home/aaydin"
APPT="apptainer exec --nv -B ${BIND} ${SIF}"

$APPT python - <<'PY'
import torch
from gsplat import rasterization

dev = torch.device("cuda")
N = 100
means = torch.randn((N, 3), device=dev)
quats = torch.nn.functional.normalize(torch.randn((N, 4), device=dev), dim=-1)
scales = torch.exp(torch.randn((N, 3), device=dev))
opacities = torch.sigmoid(torch.randn((N,), device=dev))
colors = torch.randn((N, 1, 3), device=dev)
viewmats = torch.eye(4, device=dev)[None]
Ks = torch.tensor([[[32, 0, 16], [0, 32, 16], [0, 0, 1]]], dtype=torch.float32, device=dev)

out, _, _ = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=32,
    height=32,
    sh_degree=0,
    packed=False,
)
PY

$APPT python train.py
