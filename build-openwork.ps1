param(
  [string]$TexFile = "latex\\openwork.tex",
  [ValidateSet("xelatex", "pdflatex", "lualatex")]
  [string]$Engine = "xelatex"
)

$ErrorActionPreference = "Stop"

function Ensure-Command([string]$name) {
  $cmd = Get-Command $name -ErrorAction SilentlyContinue
  if ($cmd) { return $true }

  $miktexBin = Join-Path $env:LOCALAPPDATA "Programs\\MiKTeX\\miktex\\bin\\x64"
  if (Test-Path (Join-Path $miktexBin "$name.exe")) {
    $env:PATH = "$miktexBin;$env:PATH"
    return $true
  }

  return $false
}

if (-not (Test-Path $TexFile)) {
  throw "TeX file not found: $TexFile"
}

if (-not (Ensure-Command $Engine)) {
  throw "Missing TeX engine '$Engine'. Install MiKTeX/TeX Live and ensure it's on PATH."
}

if (Ensure-Command "initexmf") {
  & initexmf --enable-installer | Out-Null
}

$texPath = (Resolve-Path $TexFile).Path
$texDir = Split-Path $texPath -Parent
$jobName = [IO.Path]::GetFileNameWithoutExtension($texPath)
$outDir = Join-Path $texDir "build"

if (-not (Test-Path $outDir)) {
  New-Item -ItemType Directory -Path $outDir | Out-Null
}

Push-Location $texDir
try {
  & $Engine -interaction=nonstopmode -file-line-error -halt-on-error -output-directory build $jobName

  if (Test-Path (Join-Path $outDir "$jobName.bcf")) {
    if (-not (Ensure-Command "biber")) {
      throw "Bibliography detected ($jobName.bcf) but 'biber' is missing. Install biber and retry."
    }
    & biber (Join-Path $outDir $jobName)
  }

  & $Engine -interaction=nonstopmode -file-line-error -halt-on-error -output-directory build $jobName
  & $Engine -interaction=nonstopmode -file-line-error -halt-on-error -output-directory build $jobName
  Write-Host "OK: $outDir\\$jobName.pdf"
}
finally {
  Pop-Location
}
