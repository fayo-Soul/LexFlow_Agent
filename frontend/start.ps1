$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-win\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到项目 Python 环境：$python"
}

Set-Location -LiteralPath $projectRoot
& $python (Join-Path $PSScriptRoot "server.py")
