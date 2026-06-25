# One-time preflight for the BGE-M3 embedding service. Run it yourself:
#   powershell -ExecutionPolicy Bypass -File services\bge-m3\setup.ps1
#
# Everything goes through uv against pyproject.toml: `uv sync` builds the venv
# (Python <3.14, cu126 torch + transformers + FastAPI), then `uv run` verifies
# CUDA and pre-downloads the BGE-M3 model so the first serve is instant.
#
# After this, run the service with:   uv run serve.py
# cu126 matches this box's CUDA 12.6 / RTX 3080; no system CUDA Toolkit needed.
# To change the CUDA build, edit the index in pyproject.toml.
#
# NB: embeddings use `transformers` directly (CLS + L2-normalize), NOT
# FlagEmbedding/sentence-transformers — those import datasets -> pyarrow, which
# segfaults on import in this environment.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=== [1/3] uv sync (venv + torch cu126 + transformers) ==="
uv sync
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "=== [2/3] verify torch / CUDA ==="
# Write the python to a temp file to avoid CLI quoting issues.
$verify = @"
import torch, fastapi, uvicorn, transformers
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} | transformers {transformers.__version__} | cuda_available: {ok}")
if ok:
    print(f"device: {torch.cuda.get_device_name(0)}")
"@
$verify | Out-File -FilePath "verify_tmp.py" -Encoding utf8
uv run python verify_tmp.py
$exit = $LASTEXITCODE
if (Test-Path "verify_tmp.py") { Remove-Item "verify_tmp.py" }
if ($exit -ne 0) { exit 1 }

Write-Host "=== [3/3] download BGE-M3 + encode smoke test (~2.3 GB first run) ==="
$smoke = @"
import torch
from transformers import AutoModel, AutoTokenizer
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3").eval().to(dev)
if dev == "cuda":
    model.half()
with torch.inference_mode():
    b = tok(["smoke test one", "smoke test two"], padding=True, truncation=True, max_length=512, return_tensors="pt")
    b = {k: v.to(dev) for k, v in b.items()}
    cls = torch.nn.functional.normalize(model(**b).last_hidden_state[:, 0], p=2, dim=1).float().cpu()
print("dense shape", tuple(cls.shape), "| device", dev, "| L2 norm[0]", round(float((cls[0] ** 2).sum() ** 0.5), 4))
"@
$smoke | Out-File -FilePath "smoke_tmp.py" -Encoding utf8
uv run python smoke_tmp.py
$exit = $LASTEXITCODE
if (Test-Path "smoke_tmp.py") { Remove-Item "smoke_tmp.py" }
if ($exit -ne 0) { exit 1 }

Write-Host ""
Write-Host "=== DONE ==="
Write-Host "Start the service:  cd services\bge-m3 ; uv run serve.py"
Write-Host "Set in .env:        BGE_M3_URL=http://localhost:8000/embed"
