# Real-Image Denoising Comparison

Experimental pipeline for comparing classical and neural image denoising methods on real smartphone datasets.

The project evaluates:

- `BM3D`
- `DnCNN`
- `FFDNet`
- `CBDNet`

Supported domains:

- `sRGB`
- `RAW`

Supported datasets:

- `SIDD Medium`
- `MIDD` for RAW evaluation

## Repository Structure

```text
benchmarking/   timing utilities
config/         experiment configuration
data/           datasets, splits, preprocessing
evaluation/     metrics and evaluation runner
models/         BM3D wrapper and neural models
notebooks/      result analysis notebook
scripts/        executable experiment entry points
training/       losses and training loop
utils/          config, logging, reproducibility, RAW helpers
```

## Environment

Use Python `3.10` or `3.11`.

`torch==2.2.1` does not provide wheels for Python `3.14`, so a newer interpreter will not work with the pinned environment. `piqa` is pinned to `1.3.2` because `0.3.3` is no longer available from the current package index.

Example setup on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Example setup on Linux or Colab:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Expected Data Layout

The default configuration expects:

```text
data/raw/SIDD_Medium_Srgb
data/raw/SIDD_Medium_Raw
data/raw/MIDD
```

These paths can be changed in [`config/experiment_config.yaml`](config/experiment_config.yaml).

The loader discovers noisy and clean pairs by filename conventions such as:

- noisy: `NOISY`, `Noisy`, `noisy`
- clean: `GT`, `CLEAN`, `Clean`, `clean`, `GROUNDTRUTH`, `GroundTruth`

For RAW data, sidecar JSON metadata may be needed for `black_level`, `white_level`, and optionally `beta1` and `beta2`.

## Experiment Workflow

Run all commands from the project root.

### 1. Verify datasets and generate deterministic splits

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config config\experiment_config.yaml --domain both
```

This checks dataset integrity and writes `data/splits.json`.

### 2. Fine-tune neural models

```powershell
.\.venv\Scripts\python.exe scripts\finetune.py --config config\experiment_config.yaml --domain both --device cuda
```

This trains `DnCNN`, `FFDNet`, and `CBDNet` and writes checkpoints to `checkpoints/`.

### 3. Evaluate BM3D and trained checkpoints

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --config config\experiment_config.yaml --domain both --device cuda
```

This writes per-image CSV files and a summary file to `results/`.

### 4. Benchmark inference speed

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_speed.py --config config\experiment_config.yaml --domain both --device cuda
```

This writes synthetic timing results to `results/benchmark_*.csv`.

## Recommended Execution Strategy

For a practical thesis workflow:

- prepare the repository and datasets locally
- run training and heavy evaluation on Google Colab Pro with GPU
- store `checkpoints/` and `results/` on Google Drive
- analyze CSV outputs in `notebooks/results_analysis.ipynb`

## Outputs

Typical generated artifacts:

- `data/splits.json`
- `checkpoints/dncnn_srgb.pt`
- `checkpoints/ffdnet_srgb.pt`
- `checkpoints/cbdnet_srgb.pt`
- `checkpoints/dncnn_raw.pt`
- `checkpoints/ffdnet_raw.pt`
- `checkpoints/cbdnet_raw.pt`
- `results/summary.csv`
- `results/benchmark_cpu.csv`
- `results/benchmark_cuda.csv`

## Notes

- `main.py` is not used as the experiment entry point; the project is driven by scripts in `scripts/`.
- Weights & Biases logging is available through the YAML config and is disabled by default.
- The repository ignores raw datasets, checkpoints, and result files by design.
