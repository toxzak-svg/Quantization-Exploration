#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorAction = "Stop"

$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Phase 0: Preparation" -ForegroundColor Cyan
Write-Host "======================================"

$LLAMA_CPP_DIR = Join-Path $PROJECT_ROOT "llama.cpp"
$MODELS_DIR = Join-Path $PROJECT_ROOT "models"
$Datasets_DIR = Join-Path $PROJECT_ROOT "data"

New-Item -ItemType Directory -Path $MODELS_DIR,$Datasets_DIR -Force | Out-Null

Write-Host "`n[1/4] Cloning llama.cpp..." -ForegroundColor Yellow
if (-not (Test-Path $LLAMA_CPP_DIR)) {
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git $LLAMA_CPP_DIR
} else {
    Write-Host "llama.cpp already exists, skipping."
}

Write-Host "`n[2/4] Downloading Llama-2-7B GGUF (Q4_K_M for baseline)..." -ForegroundColor Yellow
$MODEL_FILE = Join-Path $MODELS_DIR "llama-2-7b.q4_k_m.gguf"
if (-not (Test-Path $MODEL_FILE)) {
    $env:GIT_LFS_SKIP_SMUDGE = "1"
    git clone --depth 1 https://huggingface.co/TheBloke/Llama-2-7B-GGUF.git $MODELS_DIR\temp_llama
    Move-Item $MODELS_DIR\temp_llama\llama-2-7b.q4_k_m.gguf $MODEL_FILE
    Remove-Item -Recurse -Force $MODELS_DIR\temp_llama
} else {
    Write-Host "Model already exists, skipping."
}

Write-Host "`n[3/4] Downloading WikiText-2 dataset..." -ForegroundColor Yellow
$WIKITEXT_URL = "https://raw.githubusercontent.com/xyhuang_caltech/LLaMA/main/data/wiki.test.t.txt"
$WIKITEXT_FILE = Join-Path $Datasets_DIR "wiki.test.txt"
if (-not (Test-Path $WIKITEXT_FILE)) {
    Invoke-WebRequest -Uri $WIKITEXT_URL -OutFile $WIKITEXT_FILE
}
Write-Host "WikiText-2 downloaded."

Write-Host "`n[4/4] Installing Python dependencies..." -ForegroundColor Yellow
$PYTHON_DEPS = @("torch", "transformers", "lm-eval", "scikit-learn", "numpy")
foreach ($dep in $PYTHON_DEPS) {
    pip install $dep 2>$null
}

Write-Host "`n======================================" -ForegroundColor Green
Write-Host "Phase 0 complete!" -ForegroundColor Green
Write-Host "======================================"