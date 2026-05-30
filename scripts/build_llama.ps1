Set-StrictMode -Version Latest
$ErrorAction = "Stop"

$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$LLAMA_CPP_DIR = Join-Path $PROJECT_ROOT "llama.cpp"
$BUILD_DIR = Join-Path $LLAMA_CPP_DIR "build"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Building llama.cpp" -ForegroundColor Cyan
Write-Host "======================================"

if (-not (Test-Path $BUILD_DIR)) {
    New-Item -ItemType Directory -Path $BUILD_DIR -Force | Out-Null
}

Set-Location $LLAMA_CPP_DIR

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --config Release

$PERPLEXITY_EXE = Join-Path $BUILD_DIR "bin\Release\perplexity.exe"
if (Test-Path $PERPLEXITY_EXE) {
    Write-Host "Build successful! perplexity.exe found." -ForegroundColor Green
} else {
    Write-Host "Build completed. Check build directory for binaries." -ForegroundColor Yellow
}

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Build process finished" -ForegroundColor Cyan
Write-Host "======================================"