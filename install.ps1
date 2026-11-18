[CmdletBinding()]
param(
    [ValidateSet('Auto', 'CPU', 'CUDA')][string]$Acceleration = 'Auto',
    [switch]$CheckOnly
)

# Compatible with Windows PowerShell 5.1. Python is not required to start setup.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$script:EnvironmentPython = Join-Path $script:ProjectRoot '.venv\Scripts\python.exe'

function Invoke-PythonCode([string]$Python, [string]$Code) {
    # Base64 avoids Windows PowerShell 5.1 stripping quotes in Python -c code.
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
    & $Python -c "import base64;exec(base64.b64decode('$encoded'))"
}

function Test-CompatiblePython([string]$Candidate) {
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { return $false }
    if ($Candidate -match '\\WindowsApps\\') { return $false }
    try {
        $result = Invoke-PythonCode $Candidate 'import sys,struct,tkinter,venv; assert (3,10)<=sys.version_info[:2]<(3,13); assert struct.calcsize("P")==8; assert not hasattr(sys,"_is_gil_enabled") or sys._is_gil_enabled(); print("SOLDER_PYTHON_OK")' 2>$null
        return ($LASTEXITCODE -eq 0 -and $result -contains 'SOLDER_PYTHON_OK')
    } catch { return $false }
}

function Find-CompatiblePython {
    if (Test-CompatiblePython $script:EnvironmentPython) { return $script:EnvironmentPython }
    $candidates = [Collections.Generic.List[string]]::new()
    $candidates.Add((Join-Path $env:LOCALAPPDATA 'Programs\Python\SolderInspector312\python.exe'))
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $listing = & $launcher.Source -0p 2>$null
            foreach ($line in $listing) {
                if ($line -match '([A-Za-z]:\\.*?python(?:3(?:\.\d+)?)?\.exe)\s*$') { $candidates.Add($Matches[1]) }
            }
        } catch { }
    }
    foreach ($command in @(Get-Command python.exe, python3.exe -All -ErrorAction SilentlyContinue)) {
        $candidates.Add($command.Source)
    }
    foreach ($pattern in @(
        "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe",
        "$env:ProgramFiles\Python3*\python.exe"
    )) {
        foreach ($file in @(Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue)) { $candidates.Add($file.FullName) }
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-CompatiblePython $candidate) { return $candidate }
    }
    return $null
}

function Get-PythonInstaller {
    # Source-only patch releases are skipped: require the full x64 installer.
    $listing = Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/' -TimeoutSec 60
    $versions = [regex]::Matches($listing.Content, 'href="(3\.12\.\d+)/"') |
        ForEach-Object { [version]$_.Groups[1].Value } | Sort-Object -Descending -Unique
    foreach ($version in $versions) {
        $url = "https://www.python.org/ftp/python/$version/python-$version-amd64.exe"
        try {
            $head = Invoke-WebRequest -UseBasicParsing -Uri $url -Method Head -TimeoutSec 30
            if ($head.StatusCode -eq 200) { return [pscustomobject]@{ Version = "$version"; Url = $url } }
        } catch {
            if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 404) { continue }
            throw
        }
    }
    throw 'No supported Python x64 installer found on python.org. Check the internet connection and try again.'
}

function Get-SignedInstaller([string]$Url, [string]$Destination, [string]$Publisher) {
    Write-Host "Downloading: $Url"
    $oldProgress = $ProgressPreference
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination -TimeoutSec 600
    } finally { $ProgressPreference = $oldProgress }
    $signature = Get-AuthenticodeSignature -LiteralPath $Destination
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch $Publisher) {
        throw "Installer signature check failed: $Destination. Nothing was executed."
    }
}

function Install-ProjectPython([string]$DownloadDirectory) {
    $release = Get-PythonInstaller
    $installer = Join-Path $DownloadDirectory 'python-amd64.exe'
    Get-SignedInstaller $release.Url $installer 'Python Software Foundation'
    $target = Join-Path $env:LOCALAPPDATA 'Programs\Python\SolderInspector312'
    Write-Host "Installing Python $($release.Version) for the current Windows user..."
    $arguments = '/quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_pip=1 Include_tcltk=1 Include_test=0 Include_doc=0 Shortcuts=0 AssociateFiles=0 TargetDir="' + $target + '"'
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -notin @(0, 3010)) { throw "Python installation failed (exit $($process.ExitCode))." }
    $python = Join-Path $target 'python.exe'
    if (-not (Test-CompatiblePython $python)) { throw 'Python installation could not be verified. Restart Windows if requested, then run INSTALL.bat again.' }
    return $python
}

function Install-VCRuntime([string]$DownloadDirectory) {
    $runtime = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue
    if ($runtime -and $runtime.Installed -eq 1) { return }
    $installer = Join-Path $DownloadDirectory 'vc_redist.x64.exe'
    Get-SignedInstaller 'https://aka.ms/vs/17/release/vc_redist.x64.exe' $installer 'Microsoft Corporation'
    Write-Host 'Installing Microsoft Visual C++ Runtime. Windows may request administrator approval.'
    $process = Start-Process -FilePath $installer -ArgumentList '/install /quiet /norestart' -Verb RunAs -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -notin @(0, 3010, 1638)) { throw "Microsoft runtime installation failed (exit $($process.ExitCode))." }
    if ($process.ExitCode -eq 3010) { Write-Host 'Windows requests a restart after setup.' }
}

function Invoke-ProjectPython([string[]]$Arguments) {
    & $script:EnvironmentPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE): $($Arguments -join ' ')" }
}

function New-ProjectEnvironment([string]$Python) {
    if (Test-CompatiblePython $script:EnvironmentPython) { return }
    $environment = Join-Path $script:ProjectRoot '.venv'
    if (Test-Path -LiteralPath $environment) {
        throw 'The .venv folder exists but is unusable. Rename it to .venv-old, then run INSTALL.bat again.'
    }
    & $Python -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project Python environment.' }
    if (-not (Test-CompatiblePython $script:EnvironmentPython)) { throw 'The new Python environment could not be verified.' }
}

function Get-AccelerationMode {
    if ($Acceleration -ne 'Auto') { return $Acceleration }
    if (Test-CompatiblePython $script:EnvironmentPython) {
        try {
            Invoke-PythonCode $script:EnvironmentPython 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { return 'CUDA' }
        } catch { }
    }
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($smi) {
        $gpu = & $smi.Source --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpu) { return 'CUDA' }
    }
    return 'CPU'
}

function Test-Installation {
    if (-not (Test-CompatiblePython $script:EnvironmentPython)) { throw 'Project Python is missing or unsupported. Run INSTALL.bat first.' }
    Invoke-ProjectPython @('-m', 'pip', 'check')
    $validation = @'
import sys, tkinter, cv2, cv2_enumerate_cameras, PIL, psutil, send2trash
import rapidocr, rapidocr_onnxruntime, torch, torchvision, ultralytics
from pathlib import Path
from runtime import load_weights, load_locator_weights
root = Path.cwd()
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__, '| Ultralytics:', ultralytics.__version__)
torch.zeros(1).add_(1)
if torch.cuda.is_available():
    torch.zeros(1, device='cuda').add_(1)
    torch.cuda.synchronize()
    print('AI device: NVIDIA', torch.cuda.get_device_name(0))
else:
    print('AI device: CPU (install/update the NVIDIA driver if GPU acceleration is expected)')
window = tkinter.Tk()
window.withdraw()
window.update_idletasks()
window.destroy()
for name, loader in [('best_inspec.pt', load_weights), ('best_locator.pt', load_locator_weights)]:
    if (root / name).is_file():
        model, mapping = loader(root / name)
        print('Model OK:', name, '| classes:', len(mapping))
        del model
    else:
        print('Model missing:', name, '- copy your trained model into this folder before inspection')
print('Installation checks passed.')
'@
    Invoke-PythonCode $script:EnvironmentPython $validation
    if ($LASTEXITCODE -ne 0) { throw 'Installation verification failed. See the details above.' }
}

function Install-SolderInspector {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'This installer supports Windows x64 (Intel/AMD).'
    }
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    Push-Location -LiteralPath $script:ProjectRoot
    $transcript = $false
    $downloadDirectory = $null
    try {
        if ($CheckOnly) { Test-Installation; return }
        $logDirectory = Join-Path $script:ProjectRoot 'logs\setup'
        New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
        $logFile = Join-Path $logDirectory (('install_{0}_{1}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID))
        Start-Transcript -LiteralPath $logFile | Out-Null
        $transcript = $true
        Write-Host 'Solder Inspector - Windows setup'
        Write-Host "Project: $script:ProjectRoot"
        $downloadDirectory = Join-Path ([IO.Path]::GetTempPath()) ('solder-setup-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $downloadDirectory | Out-Null
        $python = Find-CompatiblePython
        if (-not $python) { $python = Install-ProjectPython $downloadDirectory }
        Write-Host "Python: $python"
        Install-VCRuntime $downloadDirectory
        New-ProjectEnvironment $python
        Invoke-ProjectPython @('-m', 'ensurepip', '--upgrade')
        Invoke-ProjectPython @('-m', 'pip', 'install', '--upgrade', 'pip', '--retries', '5', '--timeout', '60')
        $mode = Get-AccelerationMode
        $flavor = if ($mode -eq 'CUDA') { 'cu128' } else { 'cpu' }
        Write-Host "AI package: PyTorch 2.8.0 ($mode)"
        Invoke-ProjectPython @('-m', 'pip', 'install', '--retries', '5', '--timeout', '120', "torch==2.8.0+$flavor", "torchvision==0.23.0+$flavor", '--index-url', "https://download.pytorch.org/whl/$flavor")
        Invoke-ProjectPython @('-m', 'pip', 'install', '--retries', '5', '--timeout', '60', '-r', (Join-Path $script:ProjectRoot 'requirements-ai.txt'))
        Test-Installation
        Write-Host ''
        Write-Host 'Setup complete. Double-click start.bat to open Solder Inspector.' -ForegroundColor Green
        Write-Host "Setup log: $logFile"
    } finally {
        if ($transcript) { Stop-Transcript | Out-Null }
        if ($downloadDirectory -and (Test-Path -LiteralPath $downloadDirectory)) {
            $resolved = (Resolve-Path -LiteralPath $downloadDirectory).ProviderPath
            $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
            if ($resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
                [IO.Path]::GetFileName($resolved) -match '^solder-setup-[a-f0-9]{32}$' -and
                -not ((Get-Item -LiteralPath $resolved).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                Remove-Item -LiteralPath $resolved -Recurse -Force
            }
        }
        Pop-Location
    }
}

# Dot-source for offline tests; importing the functions never installs anything.
if ($MyInvocation.InvocationName -ne '.') {
    try { Install-SolderInspector }
    catch { Write-Host "SETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red; exit 1 }
}
