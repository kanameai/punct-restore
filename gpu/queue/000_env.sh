#!/usr/bin/env bash
# 000_env: ~/horizon/.venv を作り torch(cu126)+ライブラリを入れ、GPU を確認する。sudo なし。
set -uo pipefail
cd ~/horizon
OUT=runs/000_env
mkdir -p "$OUT"
exec > >(tee "$OUT/setup.log") 2>&1
date -u +%FT%TZ
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1 || echo "nvidia-smi failed"
python3 --version

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || { echo "venv failed (python3-venv missing?); trying virtualenv via pip --user"; 
    python3 -m pip install --user virtualenv && python3 -m virtualenv .venv; }
fi
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# torch: CUDA 12 系 wheel。cu126 がだめなら cu124 を試す
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126 \
  || python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 \
  || echo "TORCH INSTALL FAILED"

python -m pip install -r requirements.txt || echo "REQUIREMENTS INSTALL FAILED"
python -m pip freeze > "$OUT/freeze.txt"

python - <<'PY' > "$OUT/gpu_check.json"
import json, time, sys
r = {"python": sys.version}
try:
    import torch
    r["torch"] = torch.__version__
    r["torch_cuda"] = torch.version.cuda
    r["cuda_available"] = torch.cuda.is_available()
    if r["cuda_available"]:
        r["device"] = torch.cuda.get_device_name(0)
        r["mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20): c = a @ a
        torch.cuda.synchronize(); dt = time.time() - t
        r["matmul_ok"] = bool(torch.isfinite(c.float()).all())
        r["bf16_tflops"] = round(20 * 2 * 4096**3 / dt / 1e12, 1)
        x = torch.arange(6, device="cuda").float().reshape(2, 3)
        r["small_matmul"] = (x @ x.T).tolist()   # [[5,14],[14,50]]
except Exception as e:
    r["error"] = repr(e)
for m in ["transformers", "datasets", "accelerate", "huggingface_hub", "seqeval"]:
    try: r[m] = __import__(m).__version__
    except Exception as e: r[m] = "ERR " + repr(e)
print(json.dumps(r, indent=2))
PY
cat "$OUT/gpu_check.json"
df -h ~ | tail -1 > "$OUT/disk.txt"; free -g | head -2 >> "$OUT/disk.txt"
date -u +%FT%TZ
