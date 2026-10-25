param([ValidateSet('gpu', 'cpu')][string]$Device = 'gpu')
$ErrorActionPreference = 'Stop'
$env:UV_CONCURRENT_DOWNLOADS = '3'
$env:UV_HTTP_TIMEOUT = '180'
$env:NO_PROXY = (@($env:NO_PROXY, 'pypi.tuna.tsinghua.edu.cn', '.tsinghua.edu.cn') | Where-Object { $_ }) -join ','
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
# Match the user's existing Windows network settings for this process only.
if (-not $env:HTTPS_PROXY) {
    $packageUri = [uri]'https://pypi.org'
    $systemProxy = [System.Net.WebRequest]::DefaultWebProxy.GetProxy($packageUri)
    if ($systemProxy -and $systemProxy.AbsoluteUri -ne $packageUri.AbsoluteUri) {
        $env:HTTPS_PROXY = $systemProxy.AbsoluteUri
        if (-not $env:HTTP_PROXY) { $env:HTTP_PROXY = $systemProxy.AbsoluteUri }
    }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw '请先安装 uv：https://docs.astral.sh/uv/getting-started/installation/' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw '创建 Python 环境失败' }
}
$torchIndex = if ($Device -eq 'gpu') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 torchaudio==2.8.0 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch 安装失败' }
uv pip install --python .venv\Scripts\python.exe --index-url https://pypi.tuna.tsinghua.edu.cn/simple -e '.[tts,singing,test]' 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
if ($LASTEXITCODE -ne 0) { throw '音频依赖安装失败' }
Write-Host '安装完成。运行 .\start-studio.ps1，首次使用时在模型页下载所需模型。'
