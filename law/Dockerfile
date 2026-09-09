# ============================================
# 法律咨询 RAG 系统 — Docker 镜像
# ============================================
# 构建：docker build -t law-rag:latest .
# 运行：docker compose up -d
# ============================================

FROM python:3.10-slim

WORKDIR /app

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖（pymysql 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=600 --retries=10 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    -r requirements.txt

# 复制应用代码
COPY . .

# 默认端口
EXPOSE 8000

# 启动 API 服务
# 注意：大模型文件通过 volume 挂载，不在镜像中
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
