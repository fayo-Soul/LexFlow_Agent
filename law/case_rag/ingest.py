#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律案例数据入库
功能：
    1. 创建/重置 case_law.legal_case_hybrid 集合
    2. 解析 RAG/data/raw/ 下的结构化案例 .txt
    3. 对基本案情切块、生成稠密/稀疏向量
    4. 批量写入 Milvus

> 注意：setup_collection() 会删除旧集合并重建，请谨慎调用。
"""

import os
import re
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import connections, db, utility, Collection, FieldSchema, CollectionSchema, DataType
from case_rag.config import (
    MILVUS_HOST, MILVUS_PORT, DB_NAME, COLLECTION_NAME, EMBEDDING_MODEL_PATH
)
from case_rag.logger import logger


DEFAULT_RAW_DIR = os.path.join("case_rag", "data", "raw")

# 案例一级分类 -> 5 大法律领域映射
CASE_PRIMARY_CLASS_MAP = {
    "刑事": "刑法",
    "民事": "民法",
    "行政": "行政法",
    "执行": "其他法",
    "执行实施": "其他法",
    "国家赔偿": "其他法",
}

# 劳动法关键词（用于民事案例二次判断）
LABOR_KEYWORDS = [
    "劳动", "劳动合同", "工资", "工伤", "仲裁", "辞退", "解雇",
    "社保", "加班费", "经济补偿", "赔偿金", "劳务派遣", "竞业限制",
    "停工留薪", "职业病", "失业金", "生育保险"
]


def classify_case_domain(data):
    """根据一级分类和关键词将案例映射到 5 大法律领域"""
    primary = data.get("primary_class", "").strip()
    title = data.get("title", "")
    keywords = data.get("keywords", "")
    secondary = data.get("secondary_class", "")
    combined_text = f"{title} {keywords} {secondary}"

    # 先按一级分类映射
    domain = CASE_PRIMARY_CLASS_MAP.get(primary, "其他法")

    # 民事案例若命中劳动法关键词，则归为劳动法
    if domain == "民法" and any(kw in combined_text for kw in LABOR_KEYWORDS):
        domain = "劳动法"

    return domain


def setup_collection(drop_existing=True):
    """创建法律案例 Milvus 集合"""
    logger.info(f"正在连接 Milvus 服务 -> {MILVUS_HOST}:{MILVUS_PORT}")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if DB_NAME not in db.list_database():
        db.create_database(DB_NAME)
    db.using_database(DB_NAME)

    if drop_existing and utility.has_collection(COLLECTION_NAME):
        logger.info(f"检测到旧表 '{COLLECTION_NAME}'，正在删除...")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="case_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="primary_class", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="secondary_class", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="trial_stage", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="court_name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="case_number", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="index_refs", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="essence", dtype=DataType.VARCHAR, max_length=65000),
        FieldSchema(name="reason", dtype=DataType.VARCHAR, max_length=65000),
        FieldSchema(name="ruling", dtype=DataType.VARCHAR, max_length=65000),
        FieldSchema(name="fact", dtype=DataType.VARCHAR, max_length=65000),
        FieldSchema(name="child_chunk", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
    ]

    schema = CollectionSchema(fields, description="法律文书全字段冗余混合检索表")
    collection = Collection(COLLECTION_NAME, schema)
    logger.info(f"集合 '{COLLECTION_NAME}' Schema 创建成功")

    collection.create_index(
        field_name="dense_vector",
        index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
    )
    collection.create_index(
        field_name="sparse_vector",
        index_params={"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {"drop_ratio_build": 0.2}}
    )
    logger.info(f"集合 '{COLLECTION_NAME}' 索引创建完成")
    return collection


def parse_txt_file(file_path):
    """解析单个法律文书 .txt，提取标准字段"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk", errors="ignore") as f:
            content = f.read()

    if not content.strip():
        return None

    def extract_single(pattern, text, default=""):
        match = re.search(pattern, text)
        return match.group(1).strip() if match else default

    def extract_multi(pattern, text, default="", flags=re.DOTALL):
        match = re.search(pattern, text, flags)
        return match.group(1).strip() if match else default

    data = {
        "title": extract_single(r"标题：\s*(.*?)(?:\r|\n|$)", content),
        "case_id": extract_single(r"入库编号：\s*(.*?)(?:\r|\n|$)", content),
        "primary_class": extract_single(r"一级分类：\s*(.*?)(?:\r|\n|$)", content),
        "secondary_class": extract_single(r"二级分类：\s*(.*?)(?:\r|\n|$)", content),
        "trial_stage": extract_single(r"庭审：\s*(.*?)(?:\r|\n|$)", content),
        "region": extract_single(r"所属地区：\s*(.*?)(?:\r|\n|$)", content),
        "court_name": extract_single(r"法院名称：\s*(.*?)(?:\r|\n|$)", content),
        "case_number": extract_single(r"案件证号：\s*(.*?)(?:\r|\n|$)", content),
        "keywords": extract_single(r"关键词：\s*(.*?)(?:\r|\n|$)", content),
        "index_refs": extract_single(r"关联索引：\s*(.*?)(?:\r|\n|$)", content),
    }

    full_fact = extract_multi(r"基本案情：(.*?)裁决要旨：", content)
    split_match = re.search(r"([\s\S]*?)(?:刑事判决|刑事裁定|法院裁决|裁判结果|⚖️裁判结果)：([\s\S]*)", full_fact)
    if split_match:
        data["fact"] = split_match.group(1).strip()
        data["ruling"] = split_match.group(2).strip()
    else:
        data["fact"] = full_fact if full_fact else "暂无具体案情记录"
        data["ruling"] = "暂无具体刑事裁定记录"

    data["essence"] = extract_multi(r"裁决要旨：(.*?)裁判理由：", content)
    data["reason"] = extract_multi(r"裁判理由：(.*?)关键词：", content)
    data["domain"] = classify_case_domain(data)
    return data


def chunk_text(text, chunk_size=300, overlap=50):
    """滑窗切块"""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_sparse_vector(text):
    """自定义字符级稀疏向量"""
    lexical_weights = {}
    for word in list(text):
        token_id = stable_token_id(word)
        weight = 5.0 if word in ["强", "奸", "受", "贿", "判", "刑", "偷", "盗", "骗"] else 1.0
        lexical_weights[token_id] = lexical_weights.get(token_id, 0.0) + weight
    total = sum(lexical_weights.values()) if lexical_weights else 1.0
    return {int(k): float(v / total) for k, v in lexical_weights.items()}


def ingest_cases(raw_data_dir=DEFAULT_RAW_DIR, drop_existing=True):
    """完整案例入库流程"""
    if not os.path.exists(raw_data_dir):
        logger.error(f"案例原始数据目录不存在：{raw_data_dir}")
        return

    logger.info(f"正在加载 BGE-M3 模型：{EMBEDDING_MODEL_PATH}")
    # RTX 3050 4GB 显存受限，使用 fp16 可显著降低显存占用并提升吞吐
    model = SentenceTransformer(EMBEDDING_MODEL_PATH, model_kwargs={'torch_dtype': torch.float16})

    collection = setup_collection(drop_existing=drop_existing)

    all_fields = [
        "title", "case_id", "primary_class", "secondary_class", "domain", "trial_stage",
        "region", "court_name", "case_number", "keywords", "index_refs",
        "fact", "ruling", "essence", "reason", "child_chunk",
        "dense_vector", "sparse_vector"
    ]
    insert_rows = {k: [] for k in all_fields}

    files = [f for f in os.listdir(raw_data_dir) if f.endswith(".txt")]
    logger.info(f"发现 {len(files)} 个案例文件")

    # 第一阶段：解析所有案例文件并收集元数据与子块
    records = []
    all_chunks = []
    for file_name in tqdm(files, desc="解析案例文件", unit="file"):
        file_path = os.path.join(raw_data_dir, file_name)
        doc_data = parse_txt_file(file_path)
        if not doc_data:
            continue

        child_chunks = chunk_text(doc_data["fact"], chunk_size=300, overlap=50)
        if not child_chunks:
            child_chunks = [doc_data["title"][:100]]

        for chunk in child_chunks:
            records.append({"doc_data": doc_data, "chunk": chunk})
            all_chunks.append(chunk)

    total_rows = len(records)
    if total_rows == 0:
        logger.warning("没有解析到任何案例数据")
        return

    # 第二阶段：批量生成稠密向量（GPU 加速，利用 batch）
    # GPU 显存优化：RTX 3050 4GB 在 fp16 下 batch_size=4 吞吐最佳
    encode_batch_size = 4
    logger.info(f"正在为 {total_rows} 个案例子块生成向量（fp16, batch_size={encode_batch_size}, max_seq_length=512）...")
    model.max_seq_length = 512
    dense_vectors = model.encode(
        all_chunks,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    # 第三阶段：组装 Milvus 数据
    for i, record in enumerate(tqdm(records, desc="组装案例数据", unit="row")):
        doc_data = record["doc_data"]
        chunk = record["chunk"]
        for k in ["title", "case_id", "primary_class", "secondary_class", "domain", "trial_stage",
                  "region", "court_name", "case_number", "keywords", "index_refs",
                  "fact", "ruling", "essence", "reason"]:
            insert_rows[k].append(doc_data[k])
        insert_rows["child_chunk"].append(chunk)
        insert_rows["dense_vector"].append(dense_vectors[i])
        insert_rows["sparse_vector"].append(get_sparse_vector(chunk))

    total_rows = len(insert_rows["title"])
    if total_rows == 0:
        logger.warning("没有解析到任何案例数据")
        return

    logger.info(f"准备导入 {total_rows} 条案例子块数据...")

    schema_limits = {
        "case_number": 100, "case_id": 100, "primary_class": 100, "secondary_class": 100,
        "domain": 20, "trial_stage": 100, "region": 100, "title": 500, "court_name": 500,
        "keywords": 4000, "index_refs": 4000, "fact": 65000, "ruling": 65000,
        "essence": 65000, "reason": 65000, "child_chunk": 2000
    }

    def truncate_by_bytes(s, max_bytes):
        if not s:
            return ""
        encoded = str(s).encode('utf-8')
        if len(encoded) <= max_bytes:
            return s
        return encoded[:max_bytes].decode('utf-8', errors='ignore')

    milvus_data = []
    for field in collection.schema.fields:
        if field.name == "id":
            continue
        raw_column = insert_rows[field.name]
        if field.name in schema_limits:
            raw_column = [truncate_by_bytes(v, schema_limits[field.name]) for v in raw_column]
        milvus_data.append(raw_column)

    # 分批次插入 Milvus，避免单条 gRPC 消息超过 64MB 限制
    insert_batch_size = 500
    for start_idx in range(0, total_rows, insert_batch_size):
        end_idx = min(start_idx + insert_batch_size, total_rows)
        batch_data = [column[start_idx:end_idx] for column in milvus_data]
        collection.insert(batch_data)
        logger.info(f"已插入 {end_idx}/{total_rows} 条案例数据")

    collection.flush()
    logger.info(f"案例数据入库完成，共 {total_rows} 条")


if __name__ == '__main__':
    ingest_cases()
from case_rag.sparse import stable_token_id
