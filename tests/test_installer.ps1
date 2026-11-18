# Run: powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests\test_installer.ps1
$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path $PSScriptRoot -Parent) 'install.ps1')

function Assert-True($Value, [string]$Message) {
    if (-not $Value) { throw "ASSERTION FAILED: $Message" }
}

$detected = Find-CompatiblePython
Assert-True (Test-CompatiblePython $detected) 'Detect an existing supported Python with Tk and venv'
Assert-True (-not (Test-CompatiblePython 'C:\does-not-exist\python.exe')) 'Reject a missing interpreter'
Assert-True (-not (Test-CompatiblePython 'C:\Users\Nobody\AppData\Local\Microsoft\WindowsApps\python.exe')) 'Reject a Microsoft Store alias'
$quoted = Invoke-PythonCode $detected 'print("quotes and spaces survive Windows PowerShell")'
Assert-True ($quoted -eq 'quotes and spaces survive Windows PowerShell') 'Preserve quoted Python code on Windows PowerShell 5.1'

# Create and verify a real empty venv, including a project path with spaces.
& {
    $testRoot = Join-Path ([IO.Path]::GetTempPath()) ('solder-installer-test-' + [guid]::NewGuid().ToString('N'))
    $script:ProjectRoot = Join-Path $testRoot 'project with spaces'
    $script:EnvironmentPython = Join-Path $script:ProjectRoot '.venv\Scripts\python.exe'
    New-Item -ItemType Directory -Path $script:ProjectRoot | Out-Null
    try {
        New-ProjectEnvironment $detected
        Assert-True (Test-CompatiblePython $script:EnvironmentPython) 'Create a working fresh environment'
        $before = (Get-Item -LiteralPath $script:EnvironmentPython).LastWriteTimeUtc
        New-ProjectEnvironment $detected
        Assert-True ((Get-Item -LiteralPath $script:EnvironmentPython).LastWriteTimeUtc -eq $before) 'Reuse an existing valid environment'
        Write-Host 'PASS: real venv creation and reuse with spaces in project path'
    } finally {
        $resolved = (Resolve-Path -LiteralPath $testRoot).ProviderPath
        $tempPrefix = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
        if (-not $resolved.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            [IO.Path]::GetFileName($resolved) -notmatch '^solder-installer-test-[a-f0-9]{32}$') { throw 'Unsafe test cleanup path' }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}

# Test installer orchestration in an isolated directory. Network downloads,
# native installers and pip are replaced so these cases cannot change Windows.
foreach ($case in @('FreshCPU', 'ExistingCUDA', 'ModelsOnly', 'Failure')) {
    & {
        $testRoot = Join-Path ([IO.Path]::GetTempPath()) ('solder-installer-test-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $testRoot | Out-Null
        $script:ProjectRoot = $testRoot
        $script:EnvironmentPython = Join-Path $testRoot '.venv\Scripts\python.exe'
        $script:Calls = [Collections.Generic.List[string]]::new()
        $ModelsOnly = $case -eq 'ModelsOnly'
        function Find-CompatiblePython { if ($case -eq 'ExistingCUDA') { return 'mock-python.exe' }; return $null }
        function Install-ProjectPython([string]$DownloadDirectory) { $script:Calls.Add('install-python'); return 'mock-python.exe' }
        function Install-VCRuntime([string]$DownloadDirectory) { $script:Calls.Add('vc-runtime') }
        function Test-CompatiblePython([string]$Candidate) { return $true }
        function Get-AccelerationMode { if ($case -eq 'ExistingCUDA') { return 'CUDA' }; return 'CPU' }
        function Invoke-ProjectPython([string[]]$Arguments) {
            $script:Calls.Add(($Arguments -join ' '))
            if ($case -eq 'Failure') { throw 'Simulated pip failure' }
        }
        function Test-Installation { $script:Calls.Add('verified') }
        function Install-Models([string]$DownloadDirectory) { $script:Calls.Add('models') }
        $before = (Get-Location).Path
        try {
            $failed = $false
            try { Install-SolderInspector } catch { $failed = $true }
            Assert-True ($failed -eq ($case -eq 'Failure')) "$case error propagation"
            Assert-True ((Get-Location).Path -eq $before) "$case restores working directory"
            if ($case -eq 'Failure') {
                Assert-True (-not $script:Calls.Contains('verified')) 'Do not report success after a pip failure'
            } elseif ($case -eq 'ModelsOnly') {
                Assert-True ($script:Calls.Count -eq 1 -and $script:Calls.Contains('models')) 'Models-only mode does not install Python or libraries'
            } else {
                Assert-True ($script:Calls.Contains('verified')) "$case verifies after installation"
                Assert-True ($script:Calls.IndexOf('models') -ge 0 -and $script:Calls.IndexOf('models') -lt $script:Calls.IndexOf('verified')) "$case downloads models before verification"
                Assert-True (($script:Calls.Contains('install-python')) -eq ($case -eq 'FreshCPU')) "$case installs Python only when missing"
                $flavor = if ($case -eq 'ExistingCUDA') { 'cu128' } else { 'cpu' }
                Assert-True (@($script:Calls | Where-Object { $_ -match "torch==2.8.0\+$flavor" -and $_ -match "https://download.pytorch.org/whl/$flavor" }).Count -eq 1) "$case uses the requested official PyTorch build"
                Assert-True (@($script:Calls | Where-Object { $_ -match 'requirements-ai.txt' }).Count -eq 1) "$case installs application and AI requirements"
            }
            Write-Host "PASS: $case"
        } finally {
            $resolved = (Resolve-Path -LiteralPath $testRoot).ProviderPath
            $tempPrefix = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
            if (-not $resolved.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
                [IO.Path]::GetFileName($resolved) -notmatch '^solder-installer-test-[a-f0-9]{32}$') { throw 'Unsafe test cleanup path' }
            Remove-Item -LiteralPath $resolved -Recurse -Force
        }
    }
}
Write-Host 'Installer tests passed.'
