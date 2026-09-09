#!/bin/bash
# LexFlow Agent Engine - 一键启动脚本

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== LexFlow Agent Engine 启动 ==="
echo "工作目录: $PROJECT_DIR"

# 1. 环境检查
echo ""
echo "[1/4] 检查依赖..."
python3 -c "import pydantic, fastapi, uvicorn, langgraph, openai" 2>/dev/null || {
    echo "安装依赖..."
    pip install -r requirements.txt -q
}

# 2. 环境变量
echo "[2/4] 加载配置..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  已从 .env.example 创建 .env，请填入 LLM API Key"
fi

# 3. 检查 LLM Key
echo "[3/4] 验证 LLM 连接..."
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('LLM_PRIMARY_API_KEY', '')
if not key or key == '':
    print('⚠️  LLM_PRIMARY_API_KEY 为空，启动后 LLM 调用将失败')
    print('   请编辑 .env 文件填入 API Key')
else:
    print('✅  LLM Key 已配置')
" 2>/dev/null || true

# 4. 启动
echo "[4/4] 启动服务..."
echo ""
echo "    API:    http://localhost:8000"
echo "    Health: http://localhost:8000/api/v1/health"
echo "    Docs:   http://localhost:8000/docs"
echo ""

exec uvicorn lexflow_agent.api.app:app --host 0.0.0.0 --port 8000 --reload
