
param(
    [ValidateSet("Preflight", "Primary", "Shadow")]
    [string]$Mode = "Preflight",

    [int[]]$Clusters = @()
)

# MERSCOPE Agent 3 orchestration.
# Reuses frozen CosMX Agent 3 code and timing instrumentation.
# Never edits upstream data, model code, or existing run outputs.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$dataset = "merscope_brcr_validated_clean"
$base = Join-Path $root "data\pipeline_runs\$dataset"

$python = Join-Path $root ".venv\Scripts\python.exe"

$fullRunner = "scripts\agent3_validation\run_agent3_standardized_full.py"
$timedRunner = "scripts\agent3_validation\run_agent3_wallclock_timing_audit_v2.py"
$decisionRunner = "scripts\agent3_validation\run_agent3_standardized_full_with_decision.py"

$ontology = "data\resources\cell_ontology\v2026-06-08\cl-base.json"

$primaryClusters = @(2, 4, 6)
$shadowClusters = @(0, 1, 3, 5, 7)

switch ($Mode) {
    "Preflight" { $allowed = @(0, 1, 2, 3, 4, 5, 6, 7) }
    "Primary"   { $allowed = $primaryClusters }
    "Shadow"    { $allowed = $shadowClusters }
}

if ($Clusters.Count -eq 0) {
    $selected = $allowed
}
else {
    $selected = @($Clusters | Sort-Object -Unique)
}

foreach ($cluster in $selected) {
    if ($cluster -notin $allowed) {
        throw "Cluster $cluster is not permitted in mode $Mode."
    }
}

$required = @(
    $python,
    (Join-Path $root $fullRunner),
    (Join-Path $root $timedRunner),
    (Join-Path $root $decisionRunner),
    (Join-Path $root $ontology)
)

foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required frozen dependency missing: $file"
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$id = [guid]::NewGuid().ToString("N").Substring(0, 8)

Push-Location $root

try {
    foreach ($cluster in $selected) {

        $agent2 = Join-Path $base (
            "agent2_output\clusters\cluster_$cluster\" +
            "cluster_$cluster" + "_marker_evidence.json"
        )

        if (-not (Test-Path -LiteralPath $agent2 -PathType Leaf)) {
            throw "Missing Agent 2 JSON: $agent2"
        }

        $payload = Get-Content -LiteralPath $agent2 -Raw |
            ConvertFrom-Json

        # Validate the biological context before retrieval.
        if ([string]$payload.cluster_id -ne [string]$cluster) {
            throw "Cluster identity mismatch: $cluster"
        }

        if ([string]$payload.species -notin @(
            "Human", "Homo sapiens"
        )) {
            throw "Unexpected species: $($payload.species)"
        }

        if ([string]$payload.tissue -ne "Breast") {
            throw "Unexpected tissue: $($payload.tissue)"
        }

        if ([string]$payload.platform -ne "MERSCOPE") {
            throw "Unexpected platform: $($payload.platform)"
        }

        $markerCount = @($payload.input_genes).Count
        $candidateCount = @(
            $payload.ranked_candidate_cell_types
        ).Count

        if ($markerCount -lt 1 -or $markerCount -gt 50) {
            throw "Invalid marker count for C$cluster"
        }

        if ($candidateCount -lt 1) {
            throw "No Agent 2 candidates for C$cluster"
        }

        if ($Mode -eq "Preflight") {
            $outputRoot = Join-Path $base (
                "agent3_preflight\" +
                "preflight_${stamp}_${id}\cluster_$cluster"
            )
        }
        elseif ($Mode -eq "Primary") {
            $outputRoot = Join-Path $base (
                "agent3_standardized_timed\cluster_$cluster"
            )
        }
        else {
            $outputRoot = Join-Path $base (
                "agent3_shadow_timed_fixed\cluster_$cluster"
            )
        }

        # Never reuse or overwrite an existing run location.
        if (Test-Path -LiteralPath $outputRoot) {
            throw "Output already exists: $outputRoot"
        }

        Write-Host ""
        Write-Host "========================================"
        Write-Host "MERSCOPE Agent 3 | $Mode | Cluster $cluster"
        Write-Host "========================================"
        Write-Host "Markers: $markerCount"
        Write-Host "Agent 2 candidates: $candidateCount"
        Write-Host "Source: $agent2"
        Write-Host "Output: $outputRoot"

        $arguments = @(
            "--dataset", $dataset,
            "--cluster-id", [string]$cluster,
            "--disease-context", "breast cancer",
            "--agent2-json", $agent2,
            "--output-root", $outputRoot,
            "--candidate-count", "3",
            "--context-page-size", "3",
            "--rescue-page-size", "10",
            "--confirmation-page-size", "10",
            "--top-k-passages", "10",
            "--top-marker-count", "25",
            "--rescue-top-marker-limit", "5",
            "--cell-ontology", $ontology
        )

        if ($Mode -eq "Preflight") {

            # Frozen --plan-only mode:
            # build and validate requests, without retrieving papers.
            & $python -u $fullRunner @arguments "--plan-only"

            if ($LASTEXITCODE -ne 0) {
                throw "Agent 3 preflight failed for C$cluster"
            }

            Write-Host "PREFLIGHT PASS: C$cluster" -ForegroundColor Green
            continue
        }

        # Record hashes of the frozen runners and Agent 2 input.
        $protected = @(
            $agent2,
            (Join-Path $root $fullRunner),
            (Join-Path $root $timedRunner),
            (Join-Path $root $decisionRunner)
        )

        $hashesBefore = @{}

        foreach ($path in $protected) {
            $hashesBefore[$path] = (
                Get-FileHash -LiteralPath $path -Algorithm SHA256
            ).Hash
        }

        # Same timed 18-stage workflow used for CosMX.
        & $python -u $timedRunner @arguments

        if ($LASTEXITCODE -ne 0) {
            throw "Agent 3 failed for Cluster $cluster. Inspect logs."
        }

        $resultDir = Join-Path $outputRoot "result"

        $decisionPath = Join-Path $resultDir (
            "cluster_$cluster" + "_agent3_decision_summary.json"
        )

        $timingPath = Join-Path $outputRoot (
            "cluster_$cluster" + "_agent3_timing_manifest.json"
        )

        $stepwisePath = Join-Path $outputRoot (
            "cluster_$cluster" + "_agent3_stepwise_runtime.csv"
        )

        foreach ($path in @(
            $decisionPath, $timingPath, $stepwisePath
        )) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Missing required output: $path"
            }
        }

        $freezeDir = Join-Path $outputRoot "freeze"

        $freezeFiles = @(
            Get-ChildItem -LiteralPath $freezeDir `
                -Filter "*freeze_manifest.json" `
                -File -ErrorAction Stop
        )

        if ($freezeFiles.Count -ne 1) {
            throw "Expected exactly one freeze manifest for C$cluster"
        }

        $decision = Get-Content -LiteralPath $decisionPath -Raw |
            ConvertFrom-Json

        $timing = Get-Content -LiteralPath $timingPath -Raw |
            ConvertFrom-Json

        if ($decision.decision_state -notin @(
            "RESOLVED", "AMBIGUOUS", "UNRESOLVED"
        )) {
            throw "Invalid Agent 3 decision for C$cluster"
        }

        if (
            $decision.reference_label_used -eq $true -or
            $decision.expert_label_used -eq $true
        ) {
            throw "Reference/expert label leakage detected."
        }

        if ([int]$timing.return_code -ne 0) {
            throw "Agent 3 timing manifest reports failure."
        }

        foreach ($path in $protected) {
            $hashAfter = (
                Get-FileHash -LiteralPath $path -Algorithm SHA256
            ).Hash

            if ($hashAfter -ne $hashesBefore[$path]) {
                throw "Protected source changed: $path"
            }
        }

        $labels = @(
            $decision.possible_labels |
                ForEach-Object { $_.label }
        )

        $displayLabels = if ($labels.Count -gt 0) {
            $labels -join "; "
        }
        else {
            "None"
        }

        Write-Host ""
        Write-Host "========================================"
        Write-Host "VALIDATED AGENT 3 RESULT — C$cluster"
        Write-Host "========================================"
        Write-Host "Route mode   : $Mode"
        Write-Host "Decision     : $($decision.decision_state)"
        Write-Host "Labels       : $displayLabels"
        Write-Host "Runtime      : $([math]::Round($timing.total_seconds,3)) s"
        Write-Host "Freeze audit : FOUND"
        Write-Host "Source hashes: UNCHANGED"
        Write-Host "Timing CSV   : $stepwisePath"
        Write-Host "Decision JSON: $decisionPath"
        Write-Host "========================================"

        # Automatically open the scientific results in VS Code.
        code $decisionPath
        code $stepwisePath
    }
}
finally {
    Pop-Location
}
