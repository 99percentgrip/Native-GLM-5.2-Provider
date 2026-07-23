param(
    [string]$Version = $(if ($env:GLM_ACP_VERSION) { $env:GLM_ACP_VERSION } else { "latest" }),
    [string]$InstallDir = $(if ($env:GLM_ACP_INSTALL_DIR) { $env:GLM_ACP_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "Programs\NativeGLMAcp" })
)

$ErrorActionPreference = "Stop"
$repository = "99percentgrip/Native-GLM-ACP"
$releaseBase = if ($env:GLM_ACP_RELEASE_BASE_URL) { $env:GLM_ACP_RELEASE_BASE_URL.TrimEnd("/") } else { "https://github.com/$repository/releases" }

$architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
if ($architecture -ne "X64") {
    throw "glm-acp installer: unsupported Windows architecture: $architecture"
}

$asset = "native-glm-acp-windows-x86_64.zip"
if ($Version -eq "latest") {
    $downloadRoot = "$releaseBase/latest/download"
} else {
    $tag = if ($Version.StartsWith("v")) { $Version } else { "v$Version" }
    $downloadRoot = "$releaseBase/download/$tag"
}

$temporary = Join-Path ([System.IO.Path]::GetTempPath()) ("glm-acp-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temporary | Out-Null

try {
    $archive = Join-Path $temporary $asset
    $checksum = "$archive.sha256"
    Write-Host "Downloading $asset..."
    Invoke-WebRequest -Uri "$downloadRoot/$asset" -OutFile $archive
    Invoke-WebRequest -Uri "$downloadRoot/$asset.sha256" -OutFile $checksum

    $expected = ((Get-Content -Raw $checksum).Trim() -split "\s+")[0].ToUpperInvariant()
    $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToUpperInvariant()
    if ($actual -ne $expected) {
        throw "glm-acp installer: SHA-256 verification failed"
    }

    Expand-Archive -Path $archive -DestinationPath $temporary -Force
    $source = Join-Path $temporary "native-glm-acp.exe"
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "glm-acp installer: archive did not contain native-glm-acp.exe"
    }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination (Join-Path $InstallDir "native-glm-acp.exe") -Force
    Copy-Item -LiteralPath $source -Destination (Join-Path $InstallDir "glm-acp.exe") -Force

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $pathEntries = @($userPath -split ";" | Where-Object { $_ })
    if ($pathEntries -notcontains $InstallDir) {
        $updatedPath = (@($pathEntries) + $InstallDir) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $updatedPath, "User")
    }
    if (($env:Path -split ";") -notcontains $InstallDir) {
        $env:Path = "$InstallDir;$env:Path"
    }

    $installedVersion = & (Join-Path $InstallDir "native-glm-acp.exe") --version
    Write-Host "Installed Native GLM ACP ${installedVersion}:"
    Write-Host "  $(Join-Path $InstallDir 'native-glm-acp.exe')"
    Write-Host "  $(Join-Path $InstallDir 'glm-acp.exe')"
    Write-Host ""
    Write-Host "Open a new terminal, then run: glm-acp --setup"
    Write-Host "Then start the full-screen agent: glm-acp chat"
    Write-Host "Inside the TUI, type / for the live menu, paste prompts normally, and press Ctrl-X to exit."
} finally {
    Remove-Item -LiteralPath $temporary -Recurse -Force -ErrorAction SilentlyContinue
}
