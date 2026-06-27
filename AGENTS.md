# R4Tun

R4Tun is a research codebase (no web app / server): a reasoning-based multi-agent
framework that wraps the expert-designed **SAM4Tun** point-cloud segmentation
pipeline with LLM-driven parameter adaptation, for segmental tunnel-lining analysis.
Everything runs as batch CLI pipelines over point-cloud files — there are no ports,
databases, or long-running services. See `README.md` for the project overview and
`run_agents.sh` / `run_sam4tun_baseline.sh` for the pipeline entrypoints.

## Cursor Cloud specific instructions

### Environment
- Python deps live in a repo-root `venv/` (gitignored). The pipeline runner scripts
  (`run_agents.sh`, `run_sam4tun_baseline.sh`) auto-activate `venv/` if present, so keep
  the virtualenv at `/workspace/venv`. Invoke tools directly with `./venv/bin/python`.
- The startup update script already (re)creates `venv/` and installs dependencies, so
  you normally do not need to install anything by hand.
- **This VM has no GPU/CUDA.** `requirements.txt` pins `torch`/`torchvision` to the CUDA
  12.8 (Blackwell) wheels, but the update script installs the **CPU** builds
  (`torch==2.11.0+cpu`, which satisfies the `==2.11.0` pin). `torch.cuda.is_available()`
  is `False` here. The pinned `nvidia-*-cu12` wheels still get installed (they are explicit
  lines in `requirements.txt`) but are unused dead weight on this box.

### What can and cannot run here
- **Runnable on CPU (core geometry pipeline):** the early SAM4Tun stages
  `1_upfolding.py` → `2_denoising.py` → `3_enhancing.py` → `4-1_detection.py` are pure
  NumPy/SciPy/OpenCV/faiss-cpu/numba and run fine without a GPU.
- **NOT runnable here:** the SAM segmentation stage (`sam4tun/4-2_sam*.py`,
  `agents/sam.py`) needs a CUDA GPU **and** the `sam_vit_h_4b8939.pth` (~2.5 GB) checkpoint
  under `sam4tun/segment-anything/` (absent), plus `data/subsets/*.txt` tunnel data
  (only `data/sample.txt` ships in the repo). Evaluation depends on SAM output, so the
  full end-to-end run is not possible on this VM.
- The LLM "reasoning" ablations (`run_memory*.py`, `agents/*.py --ablation m|m_s|m_s_k`)
  need an API key in `.env` (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY`)
  and ultimately the SAM stage, so they cannot be exercised end-to-end here either.

### Quick hello-world (verifies the env)
Run the centre-line extraction / unwrapping stage on the bundled sample (it reads
`data/<id>.txt`, so `id=sample` maps to `data/sample.txt`; output lands in
`data/sample/unwrapped.csv`):

```bash
./venv/bin/python sam4tun/1_upfolding.py sample
```

### Gotchas
- `run_memory.py` (the default **Anthropic** orchestrator) does `import anthropic`, but
  `anthropic` is **not** in `requirements.txt`. Install it (`./venv/bin/pip install anthropic`)
  before using that script. The OpenAI (`run_memory_gpt.py`) and Gemini
  (`run_memory_gemini.py`) variants use SDKs that are already in `requirements.txt`.
- There is no test suite (`pytest` collects 0 tests) and no linter config. For a quick
  sanity check, byte-compile the scripts: `./venv/bin/python -m py_compile sam4tun/*.py agents/*.py run_*.py`.
- `data/*` is gitignored except `data/sample.txt`; pipeline outputs under `data/<id>/`
  and `data/ablation/` are scratch and will not be committed.
