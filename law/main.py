#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律问答系统统一入口

子命令：
    serve               启动交互式问答（默认）
    import-qa           解析 法律问答.txt 并导入 MySQL
    ingest-articles     把 法律条文/ 下的文档写入 Milvus（fzt_law.fzt_legal_articles）
    ingest-cases        把 case_rag/data/raw/ 下的案例写入 Milvus（fzt_case.fzt_case_hybrid）
    setup-all           依次执行 import-qa + ingest-articles + ingest-cases
    preprocess-bert     清洗 bert_balanced.jsonl
    preprocess-train    清洗 + 生成 label_vec，输出 preprocessed_data.jsonl

示例：
    python main.py
    python main.py serve
    python main.py import-qa
    python main.py ingest-articles
    python main.py ingest-cases
    python main.py setup-all
    python main.py preprocess-bert
    python main.py preprocess-train
"""

import argparse
import json
import os
import re
import sys
import time

from base import setup_logger, Config
from base.security import mask_sensitive_text
from app.citations import build_citations, build_evidence_context, validate_citations
from app.query_planner import QueryPlanner

conf = Config()
logger = setup_logger('law_main')


# ==================== 通用工具函数 ====================

def valid_source_list():
    """解析 config.ini 中的 valid_sources 字符串为列表"""
    try:
        return eval(conf.VALID_SOURCES)
    except Exception:
        return ["刑法", "民法", "劳动法", "行政法", "其他法"]


def ensure_file_exists(path):
    if not os.path.exists(path):
        logger.error(f"文件不存在：{path}")
        sys.exit(1)


def ensure_dir_exists(path):
    if not os.path.exists(path):
        logger.error(f"目录不存在：{path}")
        sys.exit(1)


# ==================== 数据预处理：法律问答对 ====================

def parse_legal_qa(input_file="法律问答.txt", output_file="law_qa_pairs.jsonl"):
    """解析 法律问答.txt 为结构化 jsonl"""
    ensure_file_exists(input_file)
    pairs = []
    current_q = None
    current_a_lines = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("问题:"):
                if current_q is not None and current_a_lines:
                    pairs.append({
                        "question": current_q.strip(),
                        "answer": "\n".join(current_a_lines).strip(),
                        "domain": ""
                    })
                current_q = line[len("问题:"):]
                current_a_lines = []
            elif line.startswith("答案:"):
                current_a_lines.append(line[len("答案:"):])
            elif current_q is not None:
                current_a_lines.append(line)

    if current_q is not None and current_a_lines:
        pairs.append({
            "question": current_q.strip(),
            "answer": "\n".join(current_a_lines).strip(),
            "domain": ""
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"解析完成：{len(pairs)} 条问答对 -> {output_file}")
    return output_file


def import_qa_to_mysql(jsonl_path="law_qa_pairs.jsonl"):
    """把 jsonl 问答对导入 MySQL"""
    ensure_file_exists(jsonl_path)
    from mysql_qa.db.law_mysql_client import LawMySQLClient

    client = LawMySQLClient()
    try:
        client.create_table()
        client.insert_data(jsonl_path)
        logger.info(f"导入完成，表中总条数：{client.get_total_count()}")
    finally:
        client.close()


# ==================== 数据预处理：BERT 训练数据清洗 ====================

def remove_adjacent_duplicates(text, min_len=5):
    if not text:
        return text
    n = len(text)
    for p in range(min_len, n // 2 + 1):
        if n % p == 0 and text == text[:p] * (n // p):
            return text[:p]
    pattern = re.compile(rf"(.{{{min_len},}})\1+")
    return pattern.sub(r"\1", text)


def clean_bert_text(text):
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\u4e00-\u9fa5\u3400-\u4dbf0-9]", "", text)
    text = remove_adjacent_duplicates(text)
    return text.strip()


def preprocess_bert(input_file="bert_balanced.jsonl", output_file="bert_balanced_cleaned.jsonl", min_len=3):
    ensure_file_exists(input_file)
    kept = 0
    dropped = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            cleaned = clean_bert_text(item.get("question", ""))
            if len(cleaned) < min_len:
                dropped += 1
                continue

            fout.write(json.dumps({"question": cleaned, "label": item.get("label", "")}, ensure_ascii=False) + "\n")
            kept += 1

    logger.info(f"BERT 数据清洗完成：保留 {kept} 条，过滤 {dropped} 条 -> {output_file}")


# ==================== 数据入库：法律条文 ====================

def ingest_articles(directory_path=".", drop_existing=True):
    """把 法律条文/ 下各法律领域文档写入 Milvus

    Args:
        directory_path: 项目根目录
        drop_existing: 是否先删除已有集合作全量重建
    """
    from rag_qa.core.document_processor import process_documents, LAW_SOURCE_MAP
    from rag_qa.core.vector_store import VectorStore

    vector_store = VectorStore()
    law_base_dir = os.path.join(directory_path, "法律条文")
    ensure_dir_exists(law_base_dir)

    valid_sources = set(valid_source_list())

    # 如需全量重建，先删除旧集合
    if drop_existing and vector_store.client.has_collection(vector_store.collection_name):
        vector_store.client.drop_collection(vector_store.collection_name)
        logger.info(f"已删除旧集合：{vector_store.collection_name}")
        vector_store._create_or_load_collection()
        logger.info(f"已重建集合：{vector_store.collection_name}")

    total_chunks = 0

    for source_name in sorted(os.listdir(law_base_dir)):
        source_dir = os.path.join(law_base_dir, source_name)
        if not os.path.isdir(source_dir):
            continue

        # 按 LAW_SOURCE_MAP 映射到 5 大法律领域后再判断是否合法
        mapped_source = LAW_SOURCE_MAP.get(source_name, source_name)
        if mapped_source not in valid_sources:
            logger.warning(f"跳过未配置的法律领域目录：{source_name}（映射后：{mapped_source}）")
            continue

        logger.info(f"正在处理法律领域：{source_name} -> {mapped_source}")
        chunks = process_documents(source_dir)
        if chunks:
            vector_store.add_documents(chunks)
            total_chunks += len(chunks)
            logger.info(f"向量数据库中添加了 {len(chunks)} 个文档块")

    logger.info(f"法律条文入库完成，共添加 {total_chunks} 个文档块")


# ==================== 数据入库：法律案例 ====================

def ingest_cases(drop_existing=True):
    """把 case_rag/data/raw/ 下的案例写入 Milvus"""
    from case_rag.ingest import ingest_cases as _ingest_cases
    _ingest_cases(drop_existing=drop_existing)


# ==================== 问答服务 ====================

def create_llm_call():
    from openai import OpenAI
    client = OpenAI(
        api_key=conf.DASHSCOPE_API_KEY,
        base_url=conf.DASHSCOPE_BASE_URL
    )

    def call_llm(prompt):
        try:
            completion = client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            return "错误：LLM返回无效内容"
        except Exception as e:
            logger.error(f"LLM API调用失败：{e}")
            return f"抱歉，处理您的问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

    return call_llm


def format_articles(article_docs):
    if not article_docs:
        return "未检索到相关法律条文。"
    return "\n\n".join([f"[{i+1}] {doc.page_content.strip()}" for i, doc in enumerate(article_docs)])


def format_cases(case_list):
    if not case_list:
        return "未检索到相关法律案例。"
    lines = []
    for i, case in enumerate(case_list, 1):
        title = case.get("title", "未知标题")
        case_number = case.get("case_number", "未知案号")
        fact = (case.get("fact", "") or "")[:500]
        ruling = (case.get("ruling", "") or "")[:500]
        lines.append(
            f"[{i}] 标题：{title}\n"
            f"    案号：{case_number}\n"
            f"    基本案情：{fact}\n"
            f"    裁判结果：{ruling}"
        )
    return "\n\n".join(lines)


class LegalResearchService:
    # 敏感关键词：命中时建议人工介入
    SENSITIVE_KEYWORDS = [
        "自杀", "自残", "家暴", "虐待", "性侵", "猥亵",
        "暴力", "故意杀人", "故意伤害致人死亡", "持枪",
        "未成年人", "儿童", "幼女",
        "巨额财产", "千万", "亿万",
        "死刑", "无期徒刑",
    ]

    def __init__(self):
        from mysql_qa.law_main import LawQASystem
        from rag_qa.core.rag_system import RAGSystem
        from rag_qa.core.vector_store import VectorStore
        from rag_qa.core.prompts import RAGPrompt
        from case_rag.searcher import LegalSearcher

        self.law_qa_system = LawQASystem()
        self.vector_store = VectorStore()
        self.llm = create_llm_call()
        self.rag_system = RAGSystem(self.vector_store, self.llm)
        self.query_planner = QueryPlanner(self.rag_system.query_classifier, self.llm)
        self.case_searcher = LegalSearcher()
        self.rag_prompt = RAGPrompt.law_rag_prompt()
        logger.info("律所法律知识库 RAG 服务初始化完成")

    @staticmethod
    def _normalize_scope(scope):
        supported = {"regulation", "judicial_interpretation", "case", "internal"}
        public_default = {"regulation", "judicial_interpretation"}
        requested = set(scope) if scope is not None else public_default
        return requested & supported

    def _check_sensitive_keywords(self, query):
        """检测问题中是否包含敏感关键词，返回匹配的关键词列表。"""
        matched = []
        for kw in self.SENSITIVE_KEYWORDS:
            if kw in query:
                matched.append(kw)
        return matched

    def search(self, query, source_filter=None, scope=None, filters=None):
        """执行法规、司法解释、案例和内部资料的统一检索。"""
        requested_scope = self._normalize_scope(scope)
        filters = filters or {}
        article_docs = []
        cases = []

        if requested_scope & {"regulation", "judicial_interpretation", "internal"}:
            document_type_filter = None
            article_scope = requested_scope & {
                "regulation", "judicial_interpretation", "internal"
            }
            if article_scope == {"internal"}:
                document_type_filter = "internal"
            article_docs = self.rag_system.retrieve_and_merge(
                query,
                source_filter=source_filter,
                strategy="直接检索",
                document_type_filter=document_type_filter,
                metadata_filters=filters,
            )

        if "case" in requested_scope:
            if article_docs:
                evidence_text = "\n\n".join(doc.page_content for doc in article_docs)
                cases = self.case_searcher.search_by_articles(
                    articles_text=evidence_text,
                    original_query=query,
                    top_k=15,
                    final_top_n=5,
                    source_filter=source_filter,
                )
            else:
                cases = self.case_searcher.hybrid_search(
                    query_text=query,
                    top_k=15,
                    final_top_n=5,
                    source_filter=source_filter,
                )

        citations = build_citations(article_docs, cases)
        return {
            "citations": citations,
            "retrieval": {
                "strategy": "hybrid",
                "retrieved_count": len(article_docs) + len(cases),
                "reranked_count": len(citations),
            },
        }

    def answer(self, query, source_filter=None, scope=None, filters=None):
        """基于可追溯证据返回法律研究辅助回答。"""
        logger.info(f"收到问题：{mask_sensitive_text(query)}")
        timing = {"classify": 0, "retrieval": 0, "llm": 0}
        t0 = time.perf_counter()
        planner = getattr(self, "query_planner", None)
        if planner is None:
            planner = QueryPlanner(
                self.rag_system.query_classifier,
                getattr(self, "llm", None),
            )
        plan = planner.plan(query, explicit_scope=scope)
        timing["classify"] = round((time.perf_counter() - t0) * 1000, 1)
        if plan.intent != "legal_research":
            return {
                "answer": plan.answer,
                "intent": plan.intent,
                "citations": [],
                "references": [],
                "used_rag": False,
                "used_fallback": False,
                "timing": timing,
                "retrieval": {
                    "strategy": "none",
                    "retrieved_count": 0,
                    "reranked_count": 0,
                },
                "classified_domain": None,
                "sensitive_keywords": [],
            }
        sensitive_keywords = self._check_sensitive_keywords(query)
        if sensitive_keywords:
            logger.warning(f"问题包含敏感关键词：{sensitive_keywords}")

        classified_domain = plan.legal_domains[0] if plan.legal_domains else None
        effective_scope = list(plan.scope)
        logger.info(
            "查询规划结果：intent=%s scope=%s domain=%s source=%s confidence=%.2f",
            plan.intent, effective_scope, classified_domain,
            plan.route_source, plan.confidence,
        )

        try:
            t_ret = time.perf_counter()
            search_result = self.search(
                query,
                source_filter=source_filter,
                scope=effective_scope,
                filters=filters,
            )
            timing["retrieval"] = round((time.perf_counter() - t_ret) * 1000, 1)
            citations = search_result["citations"]

            if not citations:
                return {
                    "answer": "当前知识库中未检索到足以支持明确结论的法律依据。建议补充案件事实、调整检索范围或由律师进一步核查。",
                    "intent": "legal_research",
                    "citations": [],
                    "references": [],
                    "used_rag": True,
                    "used_fallback": False,
                    "timing": timing,
                    "retrieval": search_result["retrieval"],
                    "classified_domain": classified_domain,
                    "sensitive_keywords": sensitive_keywords,
                }

            context = build_evidence_context(citations)
            t_llm = time.perf_counter()
            prompt = self.rag_prompt.format(
                context=context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE
            )
            answer_text = self.llm(prompt)
            timing["llm"] = round((time.perf_counter() - t_llm) * 1000, 1)
            valid, unknown = validate_citations(answer_text, citations)
            if not valid:
                logger.error(f"回答引用校验失败，未知引用：{unknown}")
                answer_text = (
                    "系统已检索到相关资料，但生成结果未通过引用一致性校验。"
                    "为避免提供无法核验的法律结论，本次不展示生成回答，请直接查看下方检索依据。"
                )

            return {
                "answer": answer_text,
                "intent": "legal_research",
                "citations": citations,
                "references": citations,
                "used_rag": True,
                "used_fallback": False,
                "timing": timing,
                "retrieval": search_result["retrieval"],
                "classified_domain": classified_domain,
                "sensitive_keywords": sensitive_keywords,
            }

        except Exception as e:
            logger.error(f"RAG 处理失败：{e}")
            return {
                "answer": "知识检索服务暂时不可用。系统不会在缺少可核验依据时生成法律结论，请稍后重试。",
                "intent": "legal_research",
                "citations": [],
                "references": [],
                "used_rag": True,
                "used_fallback": False,
                "timing": timing,
                "retrieval": {"strategy": "hybrid", "retrieved_count": 0, "reranked_count": 0},
                "classified_domain": classified_domain,
                "sensitive_keywords": sensitive_keywords,
            }

    def index_document(self, document):
        """使用现有文档处理器切分并写入法律资料 Milvus 集合。"""
        from rag_qa.core.document_processor import process_documents

        stage_dir = document["stage_dir"]
        chunks = process_documents(stage_dir)
        for chunk in chunks:
            metadata = chunk.metadata
            metadata.update({
                "document_id": document["id"],
                "chunk_id": metadata.get("id", ""),
                "document_type": document["document_type"],
                "title": document["title"],
                "version": document.get("version", "1.0"),
                "effective_status": document.get("effective_status", "unknown"),
                "access_level": document.get("access_level", "internal"),
                "lifecycle_status": document.get("lifecycle_status", "draft"),
            })
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def delete_document_index(self, document_id):
        self.vector_store.client.delete(
            collection_name=self.vector_store.collection_name,
            filter=f'document_id == "{document_id}"',
        )


# 兼容现有命令和外部导入；新代码应使用 LegalResearchService。
LawAssistant = LegalResearchService


def serve():
    assistant = LawAssistant()
    print("\n欢迎使用法律问答系统！")
    print("输入您的问题，或输入 'exit' 退出。")

    while True:
        query = input("\n请输入您的问题：").strip()
        if query.lower() == "exit":
            print("退出系统")
            break
        if not query:
            continue

        source_input = input(f"指定法律领域（可选：{'/'.join(valid_source_list())}），留空自动分类：").strip()
        source_filter = source_input if source_input else None

        result = assistant.answer(query, source_filter=source_filter)
        if isinstance(result, dict):
            print("=" * 80)
            print(f"问题：{query}")
            print(f"答案：{result['answer']}")
            print("=" * 80)
        else:
            print("=" * 80)
            print(f"问题：{query}")
            print(f"答案：{result}")
            print("=" * 80)


# ==================== 子命令入口 ====================

def cmd_import_qa(args):
    jsonl_path = parse_legal_qa()
    import_qa_to_mysql(jsonl_path)


def cmd_ingest_articles(args):
    ingest_articles(args.directory, drop_existing=args.drop_existing)


def cmd_ingest_cases(args):
    ingest_cases(drop_existing=args.drop_existing)


def cmd_setup_all(args):
    jsonl_path = parse_legal_qa()
    import_qa_to_mysql(jsonl_path)
    ingest_articles(args.directory)

    # 显式释放 ingest_articles 阶段占用的 GPU 显存与系统内存，
    # 避免后续 ingest_cases 的 SentenceTransformer 因显存碎片化/占用而降速
    import gc
    try:
        import torch
        torch.cuda.empty_cache()
        logger.info("已清空 GPU 显存缓存")
    except Exception:
        pass
    gc.collect()
    logger.info("已执行垃圾回收，准备进入案例入库阶段")

    ingest_cases(drop_existing=args.drop_existing)


def cmd_preprocess_bert(args):
    preprocess_bert(args.input, args.output)


def convert_label_to_vec(label_str):
    sources = valid_source_list()
    label_map = {name: i for i, name in enumerate(sources)}
    vec = [0] * len(sources)
    for p in label_str.replace("，", ",").split(","):
        p = p.strip()
        if p in label_map:
            vec[label_map[p]] = 1
    return vec


def preprocess_train(input_file, cleaned_file, output_file):
    preprocess_bert(input_file, cleaned_file)
    ensure_file_exists(cleaned_file)
    with open(cleaned_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            out_item = {
                "question": item.get("question", ""),
                "label_vec": convert_label_to_vec(item.get("label", ""))
            }
            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
    logger.info(f"训练数据准备完成：{output_file}")


def cmd_preprocess_train(args):
    preprocess_train(args.input, args.cleaned, args.output)


def cmd_serve(args):
    serve()


def build_parser():
    parser = argparse.ArgumentParser(
        description="法律问答系统统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py                 # 启动交互式问答
  python main.py serve           # 同上
  python main.py import-qa       # 解析并导入法律问答对到 MySQL
  python main.py ingest-articles # 把法律条文写入 Milvus
  python main.py ingest-cases    # 把法律案例写入 Milvus
  python main.py setup-all       # 一键完成所有数据准备
  python main.py preprocess-bert # 清洗 BERT 训练数据
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # serve
    subparsers.add_parser("serve", help="启动交互式问答服务")

    # import-qa
    subparsers.add_parser("import-qa", help="解析 法律问答.txt 并导入 MySQL")

    # ingest-articles
    p_articles = subparsers.add_parser("ingest-articles", help="把 法律条文/ 写入 Milvus")
    p_articles.add_argument("--directory", default=".", help="项目根目录，默认当前目录")
    p_articles.add_argument("--no-drop", action="store_true", help="不删除已存在的法律条文集合（默认全量重建）")

    # ingest-cases
    p_cases = subparsers.add_parser("ingest-cases", help="把 case_rag/data/raw/ 案例写入 Milvus")
    p_cases.add_argument("--no-drop", action="store_true", help="不删除已存在的案例集合")

    # setup-all
    p_setup = subparsers.add_parser("setup-all", help="一键完成 import-qa + ingest-articles + ingest-cases")
    p_setup.add_argument("--directory", default=".", help="项目根目录，默认当前目录")
    p_setup.add_argument("--no-drop", action="store_true", help="不删除已存在的案例集合")

    # preprocess-bert
    p_bert = subparsers.add_parser("preprocess-bert", help="清洗 BERT 训练数据")
    p_bert.add_argument("--input", default="bert_balanced.jsonl", help="输入文件")
    p_bert.add_argument("--output", default="bert_balanced_cleaned.jsonl", help="输出文件")

    # preprocess-train
    p_train = subparsers.add_parser("preprocess-train", help="清洗 BERT 数据并生成 label_vec 训练文件")
    p_train.add_argument("--input", default="bert_balanced.jsonl", help="原始输入文件")
    p_train.add_argument("--cleaned", default="bert_balanced_cleaned.jsonl", help="清洗后中间文件")
    p_train.add_argument("--output", default="preprocessed_data.jsonl", help="训练输入文件（含 label_vec）")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "serve":
        serve()
        return

    handlers = {
        "import-qa": cmd_import_qa,
        "ingest-articles": cmd_ingest_articles,
        "ingest-cases": cmd_ingest_cases,
        "setup-all": cmd_setup_all,
        "preprocess-bert": cmd_preprocess_bert,
        "preprocess-train": cmd_preprocess_train,
    }

    handler = handlers.get(args.command)
    if handler:
        # 处理 --no-drop 参数语义反转
        if hasattr(args, "no_drop"):
            args.drop_existing = not args.no_drop
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
