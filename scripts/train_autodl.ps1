$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SystemPython = if ($env:SNAKE_SYSTEM_PYTHON) {
    $env:SNAKE_SYSTEM_PYTHON
} else {
    (Get-Command python -ErrorAction Stop).Source
}

function Test-CudaTorch([string]$Python) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        return $false
    }
    & $Python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

function Test-LocalTorch([string]$Python) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        return $false
    }
    & $Python -c "import pathlib, sys, torch; prefix = pathlib.Path(sys.prefix).resolve(); torch_path = pathlib.Path(torch.__file__).resolve(); sys.exit(0 if torch_path.is_relative_to(prefix) else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

$SystemHasCudaTorch = Test-CudaTorch $SystemPython

# Let a new environment reuse an existing CUDA PyTorch installation.
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    if ($SystemHasCudaTorch) {
        Write-Host "Detected system CUDA PyTorch; creating .venv with system site packages."
        uv venv --python $SystemPython --system-site-packages ".venv"
    }
}

uv sync --no-dev --extra train --inexact

# Reuse system CUDA PyTorch in an existing environment. Remove only a local
# non-CUDA torch when it shadows the system CUDA build.
$VenvHasCudaTorch = Test-CudaTorch $VenvPython
if (-not $VenvHasCudaTorch -and $SystemHasCudaTorch) {
    Write-Host "Detected system CUDA PyTorch; enabling system site packages for the existing .venv."
    uv venv --python $SystemPython --allow-existing --system-site-packages ".venv"

    $VenvHasCudaTorch = Test-CudaTorch $VenvPython
    if (-not $VenvHasCudaTorch -and (Test-LocalTorch $VenvPython)) {
        Write-Host "Removing a non-CUDA torch from .venv because it shadows system CUDA PyTorch."
        uv pip uninstall --python $VenvPython torch
    }
}

# CPU PyTorch does not satisfy or affect the training requirement.
$VenvHasCudaTorch = Test-CudaTorch $VenvPython
if (-not $VenvHasCudaTorch) {
    Write-Host "CUDA PyTorch was not found; installing the cu124 build."
    uv sync --no-dev --extra cu124 --extra train --inexact
}

& $VenvPython -c "import sys, torch; print(f'Using torch={torch.__version__}, cuda={torch.cuda.is_available()}'); sys.exit(0 if torch.cuda.is_available() else 'GPU PyTorch is required for training')"
& $VenvPython -m snake_ai.train @args
exit $LASTEXITCODE
