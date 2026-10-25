param([switch]$SkipModels)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$revision = 'ca1e85fe9430179831e6bc6be790c332190a3866'
$runtimeRoot = Join-Path $projectRoot 'studio_data/runtimes/ace-step'
$runtimePython = Join-Path $runtimeRoot '.venv/Scripts/python.exe'
$env:UV_HTTP_TIMEOUT = '300'
$env:UV_LINK_MODE = 'copy'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:NO_PROXY = (@($env:NO_PROXY, 'pypi.tuna.tsinghua.edu.cn', '.tsinghua.edu.cn') | Where-Object { $_ }) -join ','
# Apply the existing Windows proxy only to this process. Ordinary packages use TUNA.
if (-not $env:HTTPS_PROXY) {
    $remoteUri = [uri]'https://github.com'
    $proxyUri = [System.Net.WebRequest]::DefaultWebProxy.GetProxy($remoteUri)
    if ($proxyUri -and $proxyUri.AbsoluteUri -ne $remoteUri.AbsoluteUri) {
        $env:HTTPS_PROXY = $proxyUri.AbsoluteUri
        $env:HTTP_PROXY = $proxyUri.AbsoluteUri
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $runtimeRoot '.git'))) {
    git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git $runtimeRoot
    if ($LASTEXITCODE -ne 0) { throw '自动创作源码下载失败，可重新执行续传安装。' }
}
$actualRevision = git -C $runtimeRoot rev-parse HEAD
if ($actualRevision -ne $revision) {
    git -C $runtimeRoot fetch --depth 1 origin $revision
    if ($LASTEXITCODE -ne 0) { throw '固定版本下载失败。' }
    git -C $runtimeRoot checkout --detach $revision
    if ($LASTEXITCODE -ne 0) { throw '固定版本检出失败。' }
}
if (-not (Test-Path -LiteralPath $runtimePython)) {
    uv venv --python 3.12 (Join-Path $runtimeRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw '创建独立创作环境失败。' }
}
& $runtimePython -c "import importlib.util,sys; found=importlib.util.find_spec('torch'); sys.exit(1) if found is None else None; import torch; sys.exit(0 if torch.__version__ == '2.7.1+cu128' else 1)"
if ($LASTEXITCODE -ne 0) {
    & '.venv/Scripts/python.exe' scripts/fetch_cuda_torch.py
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch 续传下载失败。' }
    $torchWheel = Join-Path $projectRoot 'work/verified-wheels/torch-2.7.1+cu128-cp312-cp312-win_amd64.whl'
    uv pip install --python $runtimePython --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match $torchWheel 'torchaudio==2.7.1+cu128' 'torchvision==0.22.1+cu128'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch 安装失败。' }
}
uv pip install --python $runtimePython --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r scripts/requirements-music.txt 'torch==2.7.1+cu128'
if ($LASTEXITCODE -ne 0) { throw '创作依赖安装失败。' }
& $runtimePython (Join-Path $PSScriptRoot 'verify-music-runtime.py') $runtimeRoot
if ($LASTEXITCODE -ne 0) { throw '独立运行环境验证失败。' }
if (-not $SkipModels) {
    & '.venv/Scripts/python.exe' -c "from pathlib import Path; from audio_studio.music_assets import download_assets; download_assets(Path('studio_data/models/ace-step').resolve())"
    if ($LASTEXITCODE -ne 0) { throw '创作权重下载失败；下次会续传。' }
}
Write-Host '自动创作环境准备完成。重新检查工作台的“歌词自动创作”状态即可使用。'
