# 律所法律知识库 RAG 系统

LawFirm Legal Knowledge RAG System

面向律师、律师助理和律所知识管理人员的法律检索与研究辅助系统。系统整合法律法规、司法解释、案例和律所内部资料，通过 Milvus 混合检索、BGE 重排序和证据约束 Prompt，生成带可核验引用的结构化回答。

本项目只负责法律知识库和 RAG，不包含多 Agent 工作流。

## 核心能力

- 法规、司法解释、案例和内部资料统一检索。
- BGE-M3 稠密向量与稀疏向量混合召回。
- BGE Reranker 二次排序。
- 自然语言法律研究问答。
- 法规、案例及原文片段引用。
- 引用编号、文档 ID 和切片 ID 一致性校验。
- 证据不足或引用异常时停止生成确定性结论。
- PDF、DOCX、TXT、Markdown 文件登记、索引和删除。
- FastAPI、OpenAPI、健康检查和基础监控。
- 默认使用少量、结构化、经过业务评测的精选法律条款，避免历史脏数据污染回答。

## 精选业务知识库

第三阶段默认启用 Milvus Collection `fzt_legal_curated_v1`，首批覆盖婚姻家事、
劳动争议、合同责任、民事诉讼时效和无证驾驶处罚。数据位于：

```text
data/curated/legal_knowledge_v1.jsonl
```

每条记录必须包含法律名称、具体条号、版本、效力状态、权威来源地址、条文原文、
业务场景和业务适用说明。重新构建命令：

```powershell
python scripts/ingest_curated_knowledge.py
```

检索质量门禁：

```powershell
python scripts/evaluate_retrieval.py
```

业务问答抽样评测：

```powershell
python scripts/evaluate_rag.py `
  --dataset data/eval/curated_business_v1.jsonl `
  --sample 3 `
  --output data/eval/curated_answer_report.json
```

默认问答只查询法规和司法解释；案例库需要用户在界面中明确选择，以免尚未完成
治理的历史案例数据污染核心法律依据。

## 产品边界

系统是律所内部法律研究辅助工具，不代替律师出具正式法律意见。大模型只能使用本次检索证据生成回答；知识库没有足够依据时，系统会明确提示证据不足，不使用模型自身知识兜底。

详细产品需求见 [产品需求文档.md](产品需求文档.md)。

## 系统流程

```text
法律问题
  -> 领域辅助识别
  -> 法规资料与案例分别检索
  -> 稠密向量 + 稀疏向量混合召回
  -> BGE Reranker
  -> 生成稳定引用编号
  -> 证据约束 Prompt
  -> 引用一致性校验
  -> 结构化研究辅助回答
```

## 技术栈

| 模块 | 技术 |
|---|---|
| API | FastAPI、Pydantic、Uvicorn |
| 向量数据库 | Milvus |
| 向量模型 | BGE-M3 |
| 重排序 | BGE Reranker |
| 大语言模型 | DashScope OpenAI 兼容接口 |
| 结构化数据 | MySQL |
| 缓存 | Redis |
| 测试 | Pytest |
| 部署 | Docker Compose |

## 快速开始

### 1. 环境要求

- Python 3.10 或更高版本
- Docker 和 Docker Compose
- 推荐 16 GB 以上内存
- 使用本地 BGE 模型时，GPU 可提升导入和检索速度

### 2. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 3. 配置

```powershell
Copy-Item .env.example .env
```

至少需要检查：

```dotenv
MYSQL_HOST=127.0.0.1
REDIS_HOST=127.0.0.1
MILVUS_HOST=127.0.0.1
DASHSCOPE_API_KEY=
API_TOKEN=
```

`.env` 可能包含密钥，禁止提交到代码仓库。生产环境必须设置 `API_TOKEN` 保护知识库管理接口。

### 4. 启动依赖

```powershell
docker compose up -d
```

如果 MySQL、Redis 和 Milvus 已部署在本地 VMware 中，无需在 Windows
上安装 Docker。将 `.env` 中对应主机统一设置为虚拟机地址，例如：

```dotenv
MYSQL_HOST=192.168.88.100
REDIS_HOST=192.168.88.100
MILVUS_HOST=192.168.88.100
```

当前 VMware 联调结果见 [docs/e2e-verification.md](docs/e2e-verification.md)。

### 5. 初始化已有数据

```powershell
python main.py ingest-articles
python main.py ingest-cases
```

也可以执行：

```powershell
python main.py setup-all
```

### 6. 启动 API

```powershell
python api_server.py
```

服务地址：

- 法律研究工作台：`http://127.0.0.1:8000/`
- OpenAPI 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/api/v1/health`
- 就绪检查：`http://127.0.0.1:8000/api/v1/ready`

模型和 Milvus 初始化需要时间。`/health` 表示 Web 服务可访问，`/ready` 表示 RAG 依赖已经初始化完成。

## 核心 API

### 第二阶段知识库工作流

内部资料采用受控生命周期：

1. `POST /api/v1/documents` 上传后状态为 `draft`。
2. `POST /api/v1/documents/{id}/lifecycle` 将状态改为 `pending_review`。
3. 管理员审核后将状态改为 `published`。
4. `POST /api/v1/documents/{id}/reindex` 将任务写入 Redis 持久化队列。
5. `GET /api/v1/index-jobs/{job_id}` 查询任务状态。
6. 已失效资料改为 `retired`，系统同时删除对应向量索引。

只有 `published` 文档允许建立索引，内部检索也只召回已发布资料。运行元数据
默认存储在 MySQL 表 `rag_documents`；首次启动会迁移原 JSON 登记记录。

Prometheus 可直接抓取 `GET /metrics`。RAG 质量门槛通过以下命令执行：

```powershell
python scripts/evaluate_rag.py --output data/eval/report.json
```

默认门槛为 Recall@K 不低于 70%、引用覆盖率不低于 90%、平均延迟不超过
15 秒且没有运行错误。任一门槛未通过时命令退出码为 2。

### 法律检索

`POST /api/v1/search`

```json
{
  "query": "违法解除劳动合同后，劳动者能否要求继续履行？",
  "scope": ["regulation", "judicial_interpretation", "case"],
  "retrieval_mode": "hybrid",
  "filters": {}
}
```

响应中的每条结果都包含 `citation_id`、`document_id`、`chunk_id`、来源位置、原文和相关性分数。

### 带引用问答

`POST /api/v1/ask`

```json
{
  "question": "未经其他股东同意对外转让股权，转让合同是否有效？",
  "scope": ["regulation", "judicial_interpretation", "case", "internal"],
  "retrieval_mode": "hybrid"
}
```

核心响应：

```json
{
  "answer": "根据检索证据……[法规1]。",
  "citations": [
    {
      "citation_id": "法规1",
      "document_id": "doc_xxx",
      "chunk_id": "chunk_xxx",
      "source_type": "regulation",
      "title": "中华人民共和国民法典",
      "location": "第五百七十七条",
      "quote": "引用原文"
    }
  ],
  "retrieval": {
    "strategy": "hybrid",
    "retrieved_count": 20,
    "reranked_count": 8
  },
  "trace_id": "请求追踪 ID"
}
```

### 上传和索引知识库文件

`POST /api/v1/documents`

请求体为文件二进制内容，查询参数：

- `file_name`：原始文件名。
- `document_type`：`regulation`、`judicial_interpretation`、`case` 或 `internal`。
- `title`：可选资料标题。

PowerShell 示例：

```powershell
$headers = @{ Authorization = "Bearer your-api-token" }
Invoke-WebRequest `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/documents?file_name=research.md&document_type=internal&title=股权转让研究" `
  -Headers $headers `
  -ContentType "application/octet-stream" `
  -InFile ".\research.md"
```

上传后调用：

```text
POST /api/v1/documents/{document_id}/reindex
```

系统会执行解析、切分、向量化并写入 Milvus。通过 `GET /api/v1/documents` 查看 `uploaded`、`indexing`、`indexed` 或 `failed` 状态。

删除接口：

```text
DELETE /api/v1/documents/{document_id}
```

删除时会先清理 Milvus 中对应 `document_id` 的向量，再删除文档登记和本地文件。

## 项目结构

```text
law/
├── app/
│   ├── citations.py          # 引用构建、上下文和校验
│   └── knowledge_base.py     # 文档登记与文件生命周期
├── base/                     # 配置、日志和安全
├── rag_qa/                   # 法规资料切分、向量化和混合检索
├── case_rag/                 # 案例入库和混合检索
├── mysql_qa/                 # 历史结构化问答数据
├── web/                      # 律师法律研究工作台
├── tests/                    # 单元与接口测试
├── api_server.py             # FastAPI 入口
├── main.py                   # RAG 服务与数据命令入口
├── 产品需求文档.md
└── docker-compose.yml
```

## 数据迁移说明

旧版本 Milvus 法规集合只保存 `source` 等基础字段。为了获得完整的 `document_id`、`chunk_id`、资料类型、标题、条款位置、版本和有效状态，升级后应重新执行法规入库。

旧数据在重新入库前仍可以被检索，但部分引用字段会由系统生成稳定替代 ID，法规条款位置和效力状态可能为空或显示为未知。

## 测试

```powershell
python -m pytest tests -q
```

测试覆盖配置安全、引用一致性、文档生命周期、API 契约、认证、健康检查和监控。

## 生产注意事项

- 必须配置管理 API Token。
- 内部案件资料发送到外部模型前，需要确认律所数据出域政策。
- 正式使用前应建立人工标注评测集，评估 `Recall@K`、引用准确率和拒答准确率。
- 法规有效状态未知时，律师必须核验现行版本。
- 大规模文件导入建议后续接入异步任务队列；当前版本的重新索引为同步执行。

本系统输出仅用于法律研究辅助，不构成正式法律意见。
