# Denoising Comparison

Experimental pipeline for comparing BM3D, DnCNN, FFDNet, and CBDNet on real smartphone denoising datasets.

## Environment

Use Python 3.10 or 3.11. The pinned thesis environment uses `torch==2.2.1`, which is not available for Python 3.14.
`piqa` is pinned to `1.3.2` because `0.3.3` is no longer available from the current pip index.

On Windows, recreate the virtual environment with Python 3.11:

```powershell
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then run scripts from the project root, for example:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config config\experiment_config.yaml --domain both
```
