param(
    [string]$TargetDir = "G:\phoenix-target-overgraph",
    [string]$ModelRoot = "G:\phoenix-models\glirel-large-v0",
    [string]$GlinerModelRoot = "C:\Users\shuga\1kittroot\1code\Angular-build\rust-native\phoenix-gliner-smoke\models\gliner-x-small",
    [switch]$PersistMemory,
    [switch]$KeepStore
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\shuga\1kittroot\1code\Angular-build\rust-native\phoenix"
$manifestPath = Join-Path $repoRoot "Cargo.toml"
$reportDir = Join-Path $repoRoot "reports\memory-smoke"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

function Invoke-CargoJson {
    param(
        [string]$Package,
        [string]$Bin,
        [string[]]$ProgramArgs
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $cargoArgs = @(
        "run",
        "--quiet",
        "--manifest-path", $manifestPath,
        "--target-dir", $TargetDir,
        "-p", $Package
    )
    if (-not [string]::IsNullOrWhiteSpace($Bin)) {
        $cargoArgs += @("--bin", $Bin)
    }
    $cargoArgs += @("--") + $ProgramArgs
    $output = & cargo @cargoArgs 2>&1
    $ErrorActionPreference = $previousPreference
    if ($LASTEXITCODE -ne 0) {
        throw "cargo run failed for package '$Package':`n$output"
    }
    $joined = ($output -join "`n")
    $firstBrace = $joined.IndexOf("{")
    $firstBracket = $joined.IndexOf("[")
    $start = if ($firstBrace -ge 0 -and ($firstBracket -lt 0 -or $firstBrace -lt $firstBracket)) {
        $firstBrace
    } else {
        $firstBracket
    }
    if ($start -lt 0) {
        throw "no json payload found for package '$Package':`n$joined"
    }
    return $joined.Substring($start)
}

Write-Host "Seeding shortrun store via phoenix-er-post..."
$erJson = Invoke-CargoJson -Package "phoenix-er-post" -Bin "" -ProgramArgs @("--corpus", "shortrun", "--keep-store", "--json", "--case-limit", "24")
$erReport = $erJson | ConvertFrom-Json
$storePath = $erReport.storePath
if ([string]::IsNullOrWhiteSpace($storePath)) {
    throw "phoenix-er-post did not return a storePath"
}

Write-Host "Running memory compiler before relation patches..."
$memoryBeforeJson = Invoke-CargoJson -Package "phoenix-memory-post" -Bin "" -ProgramArgs @("--store-path", $storePath, "--json", "--card-limit", "8")
$memoryBefore = $memoryBeforeJson | ConvertFrom-Json

Write-Host "Running relation mention seeder..."
$seedJson = Invoke-CargoJson -Package "phoenix-rel-post" -Bin "phoenix-rel-seed" -ProgramArgs @(
    "--store-path", $storePath,
    "--model-root", $GlinerModelRoot,
    "--persist-seeds",
    "--json",
    "--threshold", "0.55",
    "--chunk-size", "320",
    "--overlap", "64",
    "--max-chunks-per-archive", "8",
    "--max-windows-per-chunk", "4",
    "--max-microchunks-per-archive", "24"
)
$seedReport = $seedJson | ConvertFrom-Json

Write-Host "Running relation worker with GLiREL..."
$relArgs = @("--store-path", $storePath, "--model-root", $ModelRoot, "--persist-patches", "--json", "--case-limit", "24")
$relJson = Invoke-CargoJson -Package "phoenix-rel-post" -Bin "phoenix-rel-post" -ProgramArgs $relArgs
$relReport = $relJson | ConvertFrom-Json

Write-Host "Running memory compiler after relation patches..."
$memoryAfterArgs = @("--store-path", $storePath, "--json", "--card-limit", "8", "--persist-patches")
$memoryAfterJson = Invoke-CargoJson -Package "phoenix-memory-post" -Bin "" -ProgramArgs $memoryAfterArgs
$memoryAfter = $memoryAfterJson | ConvertFrom-Json

$summary = [pscustomobject]@{
    timestamp = $timestamp
    storePath = $storePath
    targetDir = $TargetDir
    modelRoot = $ModelRoot
    glinerModelRoot = $GlinerModelRoot
    er = $erReport
    seed = $seedReport
    relation = $relReport
    memoryBefore = $memoryBefore
    memoryAfter = $memoryAfter
}

$summaryPath = Join-Path $reportDir "shortrun-memory-loop-$timestamp.json"
$summary | ConvertTo-Json -Depth 12 | Set-Content -Path $summaryPath -Encoding UTF8

$beforeBatch = if ($memoryBefore.Count -gt 0) { $memoryBefore[0] } else { $null }
$afterBatch = if ($memoryAfter.Count -gt 0) { $memoryAfter[0] } else { $null }
$relationBatch = if ($relReport.Count -gt 0) { $relReport[0] } else { $null }

Write-Host ""
Write-Host "Store: $storePath"
Write-Host "Report: $summaryPath"
if ($beforeBatch -ne $null) {
    Write-Host ("Memory before :: claims={0} states={1} deltas={2} conflicts={3} gaps={4}" -f `
        $beforeBatch.claimCount, $beforeBatch.stateCount, $beforeBatch.deltaCount, $beforeBatch.conflictCount, $beforeBatch.gapCount)
}
if ($relationBatch -ne $null) {
    $decisionSummary = ($relationBatch.decisionCounts.PSObject.Properties | ForEach-Object { "{0}={1}" -f $_.Name, $_.Value }) -join ", "
    Write-Host ("Relation :: windows={0} cases={1} persistedRelations={2} decisions=[{3}]" -f `
        $relationBatch.windowCount, $relationBatch.reviewCaseCount, $relationBatch.persistedRelationCount, $decisionSummary)
}
if ($afterBatch -ne $null) {
    Write-Host ("Memory after  :: claims={0} states={1} deltas={2} conflicts={3} gaps={4}" -f `
        $afterBatch.claimCount, $afterBatch.stateCount, $afterBatch.deltaCount, $afterBatch.conflictCount, $afterBatch.gapCount)
}

Write-Host "Store retained for tuning: $storePath"
