# 1. 导包
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
# 进度条
from tqdm import tqdm
import torch
import numpy as np
from scipy.sparse import csr_array
from base import setup_logger, Config
# 导入os与sys模块，用于设置环境变量中路径
import os
import sys

# 2. 获取rag_qa路径 以及 project_root项目路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
# 2.1 获取rag_qa路径
rag_qa_path = os.path.dirname(current_dir)
# 2.2 把rag_qa路径添加到环境变量中，代码中寻找rag_qa_path第一时间找到（加快查询）
# 0放置于环境变量首位，rag_qa_path代表路径
sys.path.insert(0, rag_qa_path)
# 2.3 获取project_root项目路径并添加到环境变量中
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
# print(rag_qa_path)
# print(project_root)

# 3. 创建VectorStore类，用于初始化向量数据库、BGE-M3、CrossEncoder、日志对象
class VectorStore:
    def __init__(self, collection_name=Config().MILVUS_COLLECTION_NAME, host=Config().MILVUS_HOST, port=Config().MILVUS_PORT, database=Config().MILVUS_DATABASE_NAME):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        # 初始化日志
        self.logger = setup_logger('VectorStore')
        # 初始化重排序模型
        _cfg = Config()
        rerank_model_path = _cfg.BGE_RERANKER_DIR
        self.reranker = CrossEncoder(rerank_model_path, device='cpu')
        # 初始化BGE-M3嵌入模型
        bge_m3_model_path = _cfg.BGE_M3_DIR
        # 优先使用 GPU + fp16 加速嵌入；若显存不足可改回 cpu
        embed_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        use_f16 = embed_device.startswith('cuda')
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=bge_m3_model_path,
            use_f16=use_f16,
            device=embed_device
        )
        # 获取bge-m3向量维度（语义检索）
        self.dense_dim = self.embedding_function.dim['dense']
        self.logger.info(f"BGE-M3 dense dim: {self.dense_dim}")
        # 建立Milvus连接
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 调用创建或加载collection集合方法
        self._create_or_load_collection()

    # 4. 创建collection集合
    def _create_or_load_collection(self):
        if not self.client.has_collection(self.collection_name):
            # 创建schema
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 向scheam中添加字段
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # [0, 0, 0, 0.5, 0, 0, 0, 0.7]
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引
            index_params = self.client.prepare_index_params()
            # 给稠密向量添加索引
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 给稀疏向量添加索引（倒排索引）
            # 官方文档：https://docs.zilliz.com.cn/docs/use-sparse-vector
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                # drop_ratio_build代表稀疏向量的构建时删除的向量比例，默认为0.2，范围0-1，0.2表示删除20%的向量
                params={"drop_ratio_build": 0.2}
            )
            # 创建collection集合
            self.client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)
            self.logger.info(f"已成功创建集合{self.collection_name}")
        else:
            self.logger.info(f"集合{self.collection_name}已存在!")
        # 加载collection集合到内存中，确保后期可以立即查询
        self.client.load_collection(self.collection_name)

    def add_documents(self, documents, batch_size=500):
        documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if not documents:
            self.logger.warning("没有有效的文档块可处理，跳过向量生成")
            return
        # 获取所有文档内容列表
        texts = [doc.page_content for doc in documents]
        self.logger.info(f"正在为 {len(documents)} 个文档块生成向量...")

        # 分小批次编码，避免大批量时 BGE-M3 内部批处理异常
        encode_batch_size = 200
        all_dense = []
        all_sparse = []
        for start_idx in range(0, len(texts), encode_batch_size):
            batch_texts = texts[start_idx:start_idx + encode_batch_size]
            try:
                batch_emb = self.embedding_function(batch_texts)
                all_dense.extend(batch_emb["dense"])
                all_sparse.extend(batch_emb["sparse"])
            except Exception as e:
                self.logger.error(f"编码批次 {start_idx}~{start_idx + len(batch_texts)} 失败: {e}")
                # 逐条重试，跳过失败的单条
                for j, single_text in enumerate(batch_texts):
                    try:
                        single_emb = self.embedding_function([single_text])
                        all_dense.append(single_emb["dense"][0])
                        all_sparse.append(single_emb["sparse"][0])
                    except Exception:
                        all_dense.append(None)
                        all_sparse.append(None)
        # 过滤掉编码失败的条目
        valid_indices = [i for i, d in enumerate(all_dense) if d is not None]
        if not valid_indices:
            self.logger.warning("所有文档块编码均失败，跳过")
            return
        documents = [documents[i] for i in valid_indices]
        all_dense = [all_dense[i] for i in valid_indices]
        all_sparse = [all_sparse[i] for i in valid_indices]
        embeddings = {"dense": all_dense, "sparse": all_sparse}
        # 创建一个空列表data，用于存储插入的数据
        data = []  # Milvus => data=data
        for i, doc in enumerate(tqdm(documents, desc="构建 Milvus 数据", unit="chunk")):
            # 获取文档的id => 文档编码然后结合md5 + hexdigest
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # 初始化稀疏向量字典
            # 稀疏向量可能超过3万+维度，但是大部分维度为0，因此可以只保存非0的维度和值
            # "sparse_vector": {10: 0.1, 200: 0.7, 1000: 0.9}
            sparse_vector = {}
            # 获取指定文档的稀疏向量
            row = embeddings["sparse"][i]
            # 判断处理，稀疏向量有两种格式：COO风格与CSR风格
            if hasattr(row, "col"):
                # COO风格
                indices, values = row.col, row.data
            else:
                # CSR风格
                indices, values = row.indices, row.data
            # 把两个列表合并，组成字典
            # [1, 2, 3]
            # [0.1, 0.2, 0.3]
            # zip([1, 2, 3], [0.1, 0.2, 0.3])
            # 作用：相同索引进行组合 => [(1, 0.1), (2, 0.2), (3, 0.3)]
            # sparse_vector[1] = 0.1
            # sparse_vector[2] = 0.2
            # sparse_vector[3] = 0.3
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value
            # 创建数据字典，包含所有数据
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown"),
                "document_id": doc.metadata.get("document_id", ""),
                "chunk_id": doc.metadata.get("chunk_id", doc.metadata.get("id", text_hash)),
                "document_type": doc.metadata.get("document_type", "regulation"),
                "title": doc.metadata.get("title", doc.metadata.get("law_name", "")),
                "law_name": doc.metadata.get("law_name", doc.metadata.get("title", "")),
                "article_number": doc.metadata.get("article_number", ""),
                "location": doc.metadata.get("location", ""),
                "version": doc.metadata.get("version", ""),
                "effective_status": doc.metadata.get("effective_status", "unknown"),
                "access_level": doc.metadata.get("access_level", "public"),
                "lifecycle_status": doc.metadata.get("lifecycle_status", "published"),
                "source_url": doc.metadata.get("source_url", ""),
                "effective_date": doc.metadata.get("effective_date", ""),
                "promulgation_date": doc.metadata.get("promulgation_date", ""),
                "business_summary": doc.metadata.get("business_summary", ""),
                "file_path": doc.metadata.get("file_path", ""),
            })
            # 达到批次大小后执行一次批量插入
            if len(data) >= batch_size:
                self.client.upsert(collection_name=self.collection_name, data=data)
                data = []
                self.logger.info(f"已批量插入 {batch_size} 条数据")
        # 插入剩余数据
        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            self.logger.info(f"已插入剩余 {len(data)} 条数据")
        # 刷新并重新加载集合到内存
        self.client.flush(collection_name=self.collection_name)
        self.client.load_collection(self.collection_name)
        self.logger.info(f"add_documents 完成，共处理 {len(documents)} 条文档")

    # 混合检索，有3个参数，query查询问题，k代表top_k，source_filter为过滤器
    @staticmethod
    def _filter_value(value):
        """只允许预期的中文、英文、数字及常用分隔符进入 Milvus 表达式。"""
        value = str(value)
        if not value or any(char in value for char in "\"'[]()=<>"):
            raise ValueError("检索过滤条件包含非法字符")
        return value

    def hybrid_search_with_rerank(
        self,
        query,
        k=Config().RETRIEVAL_K,
        source_filter=None,
        document_type_filter=None,
        metadata_filters=None,
    ):
        # 1. query是一个自然语言，向量检索需要进行向量化处理
        query_embedding = self.embedding_function([str(query)])
        # 2 获取稠密向量数据
        dense_query_vector = query_embedding["dense"][0]
        # 3. 获取稀疏向量数据，稀疏向量可能超过3万+维度，但是大部分维度为0，因此可以只保存非0的维度和值
        sparse_query_vector = {}
        try:
            row = query_embedding["sparse"][0]
            # 判断是哪种格式，稀疏向量有两种格式：COO风格与CSR风格
            if hasattr(row, "col"):
                indices, values = row.col, row.data
            else:
                indices, values = row.indices, row.data
        except Exception as e:
            row = query_embedding["sparse"][0]
            indices, values = row.indices, row.data
        for idx, value in zip(indices, values):  # [(12,0.678),(485, 0.574),(1024, 0.6)]
            sparse_query_vector[idx] = value
        # 4. 定义filter过滤器
        # 支持单 source 字符串或 source 列表（OR 语义）
        filters = []
        if isinstance(source_filter, (list, tuple)):
            # Milvus 表达式: source in ["刑法", "民法"]
            quoted = ", ".join([f"'{self._filter_value(s)}'" for s in source_filter])
            filters.append(f"source in [{quoted}]")
        elif source_filter:
            filters.append(f"source=='{self._filter_value(source_filter)}'")

        if isinstance(document_type_filter, (list, tuple, set)):
            quoted = ", ".join(
                f"'{self._filter_value(item)}'" for item in document_type_filter
            )
            filters.append(f"document_type in [{quoted}]")
        elif document_type_filter:
            filters.append(
                f"document_type=='{self._filter_value(document_type_filter)}'"
            )
        metadata_filters = metadata_filters or {}
        effective_status = metadata_filters.get("effective_status")
        if effective_status:
            statuses = effective_status if isinstance(effective_status, list) else [effective_status]
            quoted = ", ".join(
                f"'{self._filter_value(status)}'" for status in statuses
            )
            filters.append(f"effective_status in [{quoted}]")
        access_level = metadata_filters.get("access_level")
        if access_level:
            levels = access_level if isinstance(access_level, list) else [access_level]
            quoted = ", ".join(
                f"'{self._filter_value(level)}'" for level in levels
            )
            filters.append(f"access_level in [{quoted}]")
        lifecycle_status = metadata_filters.get("lifecycle_status")
        if lifecycle_status:
            statuses = (
                lifecycle_status
                if isinstance(lifecycle_status, list)
                else [lifecycle_status]
            )
            quoted = ", ".join(
                f"'{self._filter_value(status)}'" for status in statuses
            )
            filters.append(f"lifecycle_status in [{quoted}]")
        filter_expr = " and ".join(filters)
        # 5. 基于AnnSearchRequest类进行查询检索
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )
        # 6. 定义ranker加权排序器
        # 稠密向量权重为1.0，稀疏向量权重为0.7
        ranker = WeightedRanker(1.0, 0.7)
        # 7. 创建HybridSearch类进行混合检索
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=[
                "text", "parent_id", "parent_content", "source", "timestamp",
                "document_id", "chunk_id", "document_type", "title", "law_name",
                "article_number", "location", "version", "effective_status", "file_path",
                "access_level",
                "lifecycle_status",
                "source_url", "effective_date", "promulgation_date", "business_summary",
            ]
        )[0]
        # print(results)
        # [{'id': '2b6f5a5e70906897b759640400ff59b5', 'distance': 1.1404318809509277,
        # 'entity': {'text': '测试文档1', 'parent_id': '1', 'parent_content': '测试文档1', 'source': 'test', 'timestamp': '2023-01-01'}}]
        # [{}, {}, {}]，{}都代表一个子块
        # 将搜索结果转换为Document对象 => [Document(), Document()]
        sub_chunks = []
        for hit in results:
            document = self._doc_from_hit(hit["entity"])
            document.metadata["score"] = float(hit.get("distance", 0.0))
            sub_chunks.append(document)
        self.logger.debug(f"sub_chunks count: {len(sub_chunks)}")
        # 现在获取的子块，需要遍历，获取对应父块，因为父块才是语义完整的内容，才适合喂给LLM进行回答
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # 判断，如果只有1个父块，则直接返回
        if len(parent_docs) < 2:
            return parent_docs[:Config().CANDIDATE_M]
        # 如果有多个父块，则需要使用Reranker进行排序
        if parent_docs:
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 使用BGE-Reranker计算每个配对的得分
            scores = self.reranker.predict(pairs)
            # 根据得分从高到低排序 => [(0.9, 文档内容2), (0.8, 文档内容1),]
            ranked = sorted(zip(scores, parent_docs), key=lambda item: item[0], reverse=True)
            ranked_parent_docs = []
            for score, doc in ranked:
                doc.metadata["rerank_score"] = float(score)
                ranked_parent_docs.append(doc)
        else:
            ranked_parent_docs = []
        # 返回前k个重排序后的文档
        return ranked_parent_docs[:Config().CANDIDATE_M]

    def _doc_from_hit(self, hit):
        # hit : {'parent_content': '测试文档1', 'source': 'test', 'timestamp': '2023-01-01', 'text': '测试文档1', 'parent_id': '1'}
        return Document(
            page_content=hit.get("text"),
            metadata={
                key: hit.get(key)
                for key in (
                    "parent_id", "parent_content", "source", "timestamp",
                    "document_id", "chunk_id", "document_type", "title", "law_name",
                    "article_number", "location", "version", "effective_status", "file_path",
                    "source_url", "effective_date", "promulgation_date", "business_summary",
                )
            }
        )

    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs



if __name__ == '__main__':
    vector_store = VectorStore()
    # 添加测试文档
    # vector_store.add_documents([
    #     Document(page_content="测试文档1", metadata={"parent_id": "1", "parent_content": "测试文档1", "source": "test", "timestamp": "2023-01-01"}),
    # ])
    # 混合检索
    vector_store.hybrid_search_with_rerank("测试文档1")
