$base = Split-Path -Parent $MyInvocation.MyCommand.Path

$dirs = @(
    "data\raw", "data\clips", "data\frames", "data\annotations", "data\processed",
    "models\detection", "models\tracking", "models\calibration",
    "notebooks",
    "src\video", "src\detection", "src\tracking", "src\teams",
    "src\calibration", "src\analytics", "src\visualization",
    "tests", "outputs", "scripts", "config"
)

foreach ($dir in $dirs) {
    $fullPath = Join-Path $base $dir
    New-Item -ItemType Directory -Force -Path $fullPath | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $fullPath ".gitkeep") | Out-Null
    Write-Host "Created: $dir"
}

Write-Host "`nAll directories created successfully."
