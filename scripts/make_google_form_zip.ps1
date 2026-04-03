# Build neuro-shift-google-form-upload.zip for the Google Form "Zip file" field.
# Intended when the form also has a SEPARATE ".env file" upload — this ZIP does NOT include .env.
#
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\make_google_form_zip.ps1
#
# Optional: include .env inside the ZIP as well (e.g. single-zip-only forms):
#   powershell -ExecutionPolicy Bypass -File scripts\make_google_form_zip.ps1 -IncludeEnv

param([switch]$IncludeEnv)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$readme = Join-Path $root "FORM_UPLOAD_README.txt"
$eval = Join-Path $root "EVALUATOR_QUICKSTART.txt"
$envFile = Join-Path $root ".env"
$zipOut = Join-Path $root "neuro-shift-google-form-upload.zip"

if (-not (Test-Path $readme)) {
    Write-Host "ERROR: FORM_UPLOAD_README.txt missing from repo root." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $eval)) {
    Write-Host "ERROR: EVALUATOR_QUICKSTART.txt missing from repo root." -ForegroundColor Red
    exit 1
}
if ($IncludeEnv -and -not (Test-Path $envFile)) {
    Write-Host "ERROR: .env not found (use -IncludeEnv only after creating .env)." -ForegroundColor Red
    exit 1
}

if (Test-Path $zipOut) { Remove-Item $zipOut -Force }

if ($IncludeEnv) {
    Compress-Archive -LiteralPath $eval, $readme, $envFile -DestinationPath $zipOut -CompressionLevel Optimal -Force
    Write-Host "ZIP includes .env (for forms without a separate .env field)." -ForegroundColor Yellow
} else {
    Compress-Archive -LiteralPath $eval, $readme -DestinationPath $zipOut -CompressionLevel Optimal -Force
}

Write-Host "Created: $zipOut" -ForegroundColor Green
Write-Host ""
Write-Host "Google Form:" -ForegroundColor Cyan
Write-Host "  Zip file field  -> upload this ZIP"
Write-Host "  .env field      -> upload: $envFile"
Write-Host "  (If .env missing, copy .env.example to .env and fill Tuya values first.)"
Write-Host ""
Write-Host "Do not commit this ZIP or .env to GitHub."
