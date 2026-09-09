# LexFlow Agent

LexFlow Agent 是面向劳动争议案件的四 Agent LangGraph 服务。

## 架构

1. **文件解析工具**：自动将 PDF/DOCX/TXT/MD/图片解析为分页纯文本（PDF/图片走 MinerU，DOCX 走 python-docx）。
2. 案件理解 Agent：多材料并行评审与事实来源校验。
3. 法律研究 Agent：多查询 Law RAG 与确定性引用门禁。
4. 案件策略 Agent：证据矩阵、主备策略与人工审核。
5. 文书生成 Agent：状态机、质量复核、定向回退与 SSE。

Orchestrator 负责四张子图编排、SQLite checkpoint、人工暂停、最大修改次数和失败转人工。

## 运行

```bash
# 安装 MinerU（PDF/图片解析，需独立 conda 环境）
conda create -n mineru python=3.10 && conda activate mineru
pip install -U "mineru[all]" --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple

# 安装项目依赖
pip install -r requirements.txt

docker compose up -d --build
```

- Agent API：`http://localhost:8000`
- Law RAG：`http://localhost:8001`
- OpenAPI：`http://localhost:8000/docs`

除健康接口外，案件接口均要求：

```text
Authorization: Bearer <API_TOKEN>
```

## 文件上传

`POST /api/v1/chat/stream` 支持通过 `file_paths` 字段传入原始文件路径：

```json
{
  "message": "分析这个劳动争议案件",
  "file_paths": ["/uploads/劳动合同.pdf", "/uploads/解除通知.docx"]
}
```

文件解析在路由判决之前完成，支持 PDF/DOCX/TXT/MD/PNG/JPG 格式。解析后的分页文本自动注入 Agent Pipeline。

配置项见 `.env` 中 `MINERU_*` / `FILE_PARSER_ENABLED` 等。

## LangSmith

- Trace / Run：LangGraph 自动追踪，并附带 `thread_id`、案件 ID、部署版本。
- Tool Call：Law RAG 的 search/ask 使用 tool run。
- Model Call：OpenAI 兼容客户端由 LangSmith wrapper 记录。
- Prompt / Agent：`scripts/setup_langsmith.py` 同步 Prompt；四张子图作为 Agent 节点。
- Dataset / Eval：30 个合成脱敏案件；`scripts/run_langsmith_evaluation.py` 创建实验。
- Thread：每个 API `run_id` 同时作为 LangSmith `thread_id`。
- Feedback / Annotation：`POST /api/v1/case-runs/{run_id}/feedback`，低分自动入审核队列。
- Deployment：根目录 `langgraph.json` 声明 `lexflow-orchestrator`。

## 验证

```bash
pytest -m "needs_conda" -v
python -m pytest -q
python scripts/run_evaluation.py
```

生产部署前必须配置非默认 `API_TOKEN`、LangSmith API Key 和 LLM API Key。
