[CmdletBinding()]
param(
    [ValidateSet('Auto', 'CPU', 'CUDA')][string]$Acceleration = 'Auto',
    [switch]$CheckOnly,
    [switch]$ModelsOnly
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

function Get-ModelManifest {
    $path = Join-Path $script:ProjectRoot 'models.json'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw 'models.json is missing. Download the complete project folder and run INSTALL.bat again.' }
    $manifest = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.schema_version -ne 1) { throw 'Unsupported models.json format.' }
    $expected = @('best_inspec.pt', 'best_locator.pt')
    $files = @($manifest.files)
    if ($files.Count -ne 2 -or @(Compare-Object $expected @($files | ForEach-Object { $_.name })).Count -ne 0) {
        throw 'models.json must contain best_inspec.pt and best_locator.pt exactly once.'
    }
    foreach ($model in $files) {
        $url = $null
        if (-not [Uri]::TryCreate([string]$model.url, [UriKind]::Absolute, [ref]$url) -or $url.Scheme -ne 'https' -or $url.UserInfo) {
            throw "Model download must use an HTTPS link without credentials: $($model.name)"
        }
        if ([string]$model.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or [long]$model.size -le 0 -or [long]$model.size -gt 2GB) {
            throw "Invalid expected model size or SHA256: $($model.name)"
        }
    }
    return $files
}

function Receive-ModelFile([string]$Url, [string]$Destination, [long]$ExpectedSize) {
    # Stream large weights directly to disk. Windows PowerShell 5.1's web
    # cmdlet can buffer a large response and appear stalled with zero bytes.
    Add-Type -AssemblyName System.Net.Http
    $client = [Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds(600)
    $client.DefaultRequestHeaders.UserAgent.ParseAdd('SolderInspector-Installer/1.0')
    $cancel = [Threading.CancellationTokenSource]::new()
    $cancel.CancelAfter(600000)
    $response = $null
    $inputStream = $null
    $outputStream = $null
    try {
        $response = $client.GetAsync($Url, [Net.Http.HttpCompletionOption]::ResponseHeadersRead, $cancel.Token).GetAwaiter().GetResult()
        $response.EnsureSuccessStatusCode() | Out-Null
        $inputStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $outputStream = [IO.File]::Open($Destination, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $buffer = [byte[]]::new(1MB)
        [long]$received = 0
        [long]$nextNotice = 16MB
        while ($true) {
            $count = $inputStream.ReadAsync($buffer, 0, $buffer.Length, $cancel.Token).GetAwaiter().GetResult()
            if ($count -eq 0) { break }
            $outputStream.Write($buffer, 0, $count)
            $received += $count
            if ($received -gt $ExpectedSize) { throw 'Download is larger than the expected model size.' }
            if ($received -ge $nextNotice) {
                Write-Host ('  {0:N1} / {1:N1} MiB' -f ($received / 1MB), ($ExpectedSize / 1MB))
                $nextNotice = $received + 16MB
            }
        }
    } finally {
        if ($outputStream) { $outputStream.Dispose() }
        if ($inputStream) { $inputStream.Dispose() }
        if ($response) { $response.Dispose() }
        $client.Dispose()
        $cancel.Dispose()
    }
}

function Install-Models([string]$DownloadDirectory) {
    foreach ($model in @(Get-ModelManifest)) {
        $destination = Join-Path $script:ProjectRoot $model.name
        if (Test-Path -LiteralPath $destination) {
            if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) { throw "A directory is blocking model installation: $destination" }
            $size = (Get-Item -LiteralPath $destination).Length
            if ($size -eq [long]$model.size -and (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -eq $model.sha256) {
                Write-Host "Model already installed: $($model.name)"
            } else {
                # The operator may have replaced the release with newer trained
                # weights. Preserve them; Test-Installation validates their use.
                Write-Host "Keeping existing custom model: $($model.name). Rename it if you want to download the bundled model."
            }
            continue
        }
        # Stage beside the destination so the final move stays on one volume.
        $partial = Join-Path $script:ProjectRoot ('.model-download-' + [guid]::NewGuid().ToString('N') + '.pt')
        $verified = $false
        try {
            for ($attempt = 1; $attempt -le 3; $attempt++) {
                Write-Host ('Downloading {0} ({1:N1} MiB), attempt {2}/3...' -f $model.name, ([long]$model.size / 1MB), $attempt)
                $oldProgress = $ProgressPreference
                try {
                    $ProgressPreference = 'SilentlyContinue'
                    Receive-ModelFile $model.url $partial ([long]$model.size)
                    if ((Get-Item -LiteralPath $partial).Length -ne [long]$model.size) { throw 'Downloaded file size does not match the manifest.' }
                    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $model.sha256) { throw 'Downloaded file SHA256 does not match the manifest.' }
                    $verified = $true
                    break
                } catch {
                    if ($attempt -eq 3) { throw "Could not download $($model.name). Check the internet connection and run INSTALL.bat again. $($_.Exception.Message)" }
                    Write-Host "Download incomplete; retrying $($model.name)..."
                } finally { $ProgressPreference = $oldProgress }
            }
            if (-not $verified) { throw "Model verification failed: $($model.name)" }
            # Install only a complete verified file. Do not overwrite a local model
            # if another program created it while the download was in progress.
            Move-Item -LiteralPath $partial -Destination $destination -ErrorAction Stop
            Write-Host "Model verified and installed: $($model.name)"
        } finally {
            if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
        }
    }
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
        raise FileNotFoundError(f'Model missing: {name}. Run INSTALL.bat to download the models.')
print('Installation checks passed.')
'@
    Invoke-PythonCode $script:EnvironmentPython $validation
    if ($LASTEXITCODE -ne 0) { throw 'Installation verification failed. See the details above.' }
}

function Install-SolderInspector {
    if ($CheckOnly -and $ModelsOnly) { throw 'Choose either -CheckOnly or -ModelsOnly.' }
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
        if ($ModelsOnly) {
            Install-Models $downloadDirectory
            Write-Host 'Model files are ready.' -ForegroundColor Green
            return
        }
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
        Install-Models $downloadDirectory
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
