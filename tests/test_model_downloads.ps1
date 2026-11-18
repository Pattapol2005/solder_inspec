# Offline tests for model download integrity, preservation and failure handling.
$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path $PSScriptRoot -Parent) 'install.ps1')
function Assert-True($Value, [string]$Message) {
    if (-not $Value) { throw "ASSERTION FAILED: $Message" }
}
$manifestRoot = $script:ProjectRoot
Assert-True (@(Get-ModelManifest).Count -eq 2) 'The shipped manifest is valid'

foreach ($case in @('Download', 'Existing', 'Custom', 'Retry', 'BadHash', 'BadSize', 'Network', 'BadName', 'BadURL', 'Duplicate')) {
    & {
        $testRoot = Join-Path ([IO.Path]::GetTempPath()) ('solder-model-test-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $testRoot | Out-Null
        $script:ProjectRoot = $testRoot
        $script:DownloadCalls = 0
        $contents = [Text.Encoding]::UTF8.GetBytes('fake model content for transport tests')
        $sha = [Security.Cryptography.SHA256]::Create()
        try { $hash = ([BitConverter]::ToString($sha.ComputeHash($contents))).Replace('-', '').ToLowerInvariant() }
        finally { $sha.Dispose() }
        $models = @('best_inspec.pt', 'best_locator.pt') | ForEach-Object {
            [pscustomobject]@{ name = $_; url = "https://example.com/$_"; size = $contents.Length; sha256 = $hash }
        }
        if ($case -eq 'BadName') { $models[0].name = '..\outside.pt' }
        if ($case -eq 'BadURL') { $models[0].url = 'http://example.com/unsafe.pt' }
        if ($case -eq 'Duplicate') { $models[1].name = 'best_inspec.pt' }
        @{ schema_version = 1; files = @($models) } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $testRoot 'models.json') -Encoding UTF8
        if ($case -in @('Existing', 'Custom')) {
            foreach ($name in @('best_inspec.pt', 'best_locator.pt')) {
                $bytes = if ($case -eq 'Custom') { [Text.Encoding]::UTF8.GetBytes('user supplied weights') } else { $contents }
                [IO.File]::WriteAllBytes((Join-Path $testRoot $name), $bytes)
            }
        }
        # Replace HTTP only; file verification and final same-volume rename are real.
        function Receive-ModelFile {
            param([string]$Url, [string]$Destination, [long]$ExpectedSize)
            $script:DownloadCalls++
            if ($case -eq 'Network') { throw 'Simulated offline network' }
            if ($case -eq 'BadHash' -or ($case -eq 'Retry' -and $script:DownloadCalls -eq 1)) {
                $bad = [byte[]]$contents.Clone()
                $bad[0] = 0
                [IO.File]::WriteAllBytes($Destination, $bad)
            } elseif ($case -eq 'BadSize') {
                [IO.File]::WriteAllBytes($Destination, [byte[]]@(1, 2))
            } else { [IO.File]::WriteAllBytes($Destination, $contents) }
        }
        try {
            $failed = $false
            try { Install-Models $testRoot } catch { $failed = $true }
            $expectedFailure = $case -in @('BadHash', 'BadSize', 'Network', 'BadName', 'BadURL', 'Duplicate')
            Assert-True ($failed -eq $expectedFailure) "$case failure state"
            if ($expectedFailure) {
                Assert-True (-not (Test-Path -LiteralPath (Join-Path $testRoot 'best_inspec.pt'))) "$case never publishes invalid data"
                if ($case -in @('BadHash', 'BadSize', 'Network')) {
                    Assert-True ($script:DownloadCalls -eq 3) "$case stops after three attempts"
                } else { Assert-True ($script:DownloadCalls -eq 0) "$case is rejected before downloading" }
            } elseif ($case -eq 'Custom') {
                Assert-True ($script:DownloadCalls -eq 0) 'Do not download over custom weights'
                Assert-True ([IO.File]::ReadAllText((Join-Path $testRoot 'best_inspec.pt')) -eq 'user supplied weights') 'Preserve custom model bytes'
            } else {
                foreach ($name in @('best_inspec.pt', 'best_locator.pt')) {
                    Assert-True ((Get-FileHash -LiteralPath (Join-Path $testRoot $name) -Algorithm SHA256).Hash -eq $hash) "$case installs correct bytes"
                }
                if ($case -eq 'Existing') { Assert-True ($script:DownloadCalls -eq 0) 'Valid existing models skip network' }
                if ($case -eq 'Retry') { Assert-True ($script:DownloadCalls -eq 3) 'A corrupt first attempt can recover' }
            }
            Assert-True (@(Get-ChildItem -LiteralPath $testRoot -Filter '.model-download-*' -Force).Count -eq 0) "$case cleans partial files"
            Write-Host "PASS: $case"
        } finally {
            $resolved = (Resolve-Path -LiteralPath $testRoot).ProviderPath
            $tempPrefix = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
            if (-not $resolved.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
                [IO.Path]::GetFileName($resolved) -notmatch '^solder-model-test-[a-f0-9]{32}$') { throw 'Unsafe test cleanup path' }
            Remove-Item -LiteralPath $resolved -Recurse -Force
        }
    }
}
$script:ProjectRoot = $manifestRoot
Write-Host 'Model download tests passed.'
