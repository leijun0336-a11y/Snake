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

function Test-TorchImport([string]$Python) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        return $false
    }
    & $Python -c "import torch" *> $null
    return $LASTEXITCODE -eq 0
}

$SystemHasTorch = Test-TorchImport $SystemPython

# Let a new environment reuse an existing CPU or CUDA PyTorch installation.
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    if ($SystemHasTorch) {
        Write-Host "Detected system PyTorch; creating .venv with system site packages."
        uv venv --python $SystemPython --system-site-packages ".venv"
    }
}

uv sync --inexact

# Also handle an existing .venv that cannot see system site packages yet.
$VenvHasTorch = Test-TorchImport $VenvPython
if (-not $VenvHasTorch -and $SystemHasTorch) {
    Write-Host "Detected system PyTorch; enabling system site packages for the existing .venv."
    uv venv --python $SystemPython --allow-existing --system-site-packages ".venv"
    $VenvHasTorch = Test-TorchImport $VenvPython
}

if (-not $VenvHasTorch) {
    Write-Host "PyTorch was not found; installing the CPU build."
    uv sync --extra cpu --inexact
} else {
    & $VenvPython -c "import torch; print(f'Using torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
}

& $VenvPython -m snake_ai.evaluate @args
exit $LASTEXITCODE
