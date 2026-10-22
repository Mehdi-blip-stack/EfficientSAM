<#+
.SYNOPSIS
Wait for the official Oxford-IIIT Pets image extraction, then run raw-data EDA.

.DESCRIPTION
This helper is intentionally limited to the EDA phase. It does not resize,
cache, normalize, augment, or train. It waits until the official image folder
contains the 7,349 image IDs named in the supplied manifests, then invokes
eda_pets.py with the project's Python environment.
#>

[CmdletBinding()]
param(
    [ValidateRange(1, [int]::MaxValue)]
    [int]$ExpectedManifestImages = 7349,
    [ValidateRange(1, [int]::MaxValue)]
    [int]$PollSeconds = 60
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$rawRoot = Join-Path $projectRoot 'data\raw'
$imagesRoot = Join-Path $rawRoot 'images'
$annotationsRoot = Join-Path $rawRoot 'annotations'
$trimapsRoot = Join-Path $annotationsRoot 'trimaps'
$manifestPaths = @(
    (Join-Path $annotationsRoot 'trainval.txt'),
    (Join-Path $annotationsRoot 'test.txt')
)
$python = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$edaScript = Join-Path $PSScriptRoot 'eda_pets.py'
$outputRoot = Join-Path $projectRoot 'runs\eda\oxford_iiit_pets'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Expected Python environment was not found: $python"
}

while ($true) {
    $missingPrerequisites = @(
        @($imagesRoot, $trimapsRoot) + $manifestPaths | Where-Object {
            -not (Test-Path -LiteralPath $_)
        }
    )
    if ($missingPrerequisites.Count -gt 0) {
        Write-Output "$(Get-Date -Format s) waiting for raw-data folders or manifests: $($missingPrerequisites -join ', ')"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $expectedImageIds = @(
        foreach ($manifestPath in $manifestPaths) {
            Get-Content -LiteralPath $manifestPath | ForEach-Object {
                $line = $_.Trim()
                if ($line -and -not $line.StartsWith('#')) {
                    ($line -split '\s+')[0]
                }
            }
        }
    )
    if ($expectedImageIds.Count -ne $ExpectedManifestImages) {
        throw "Expected $ExpectedManifestImages manifest rows, but found $($expectedImageIds.Count)."
    }

    $missingImages = @(
        $expectedImageIds | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $imagesRoot ([string]::Concat($_, '.jpg'))) -PathType Leaf)
        }
    )
    $missingTrimaps = @(
        $expectedImageIds | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $trimapsRoot ([string]::Concat($_, '.png'))) -PathType Leaf)
        }
    )
    $readyImages = $expectedImageIds.Count - $missingImages.Count
    $readyTrimaps = $expectedImageIds.Count - $missingTrimaps.Count
    Write-Output "$(Get-Date -Format s) waiting for expected raw pairs: images $readyImages/$ExpectedManifestImages; trimaps $readyTrimaps/$ExpectedManifestImages"
    if ($missingImages.Count -eq 0 -and $missingTrimaps.Count -eq 0) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

& $python $edaScript --root $rawRoot --out $outputRoot
exit $LASTEXITCODE
