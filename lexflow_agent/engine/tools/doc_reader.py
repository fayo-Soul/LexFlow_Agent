"""案件文档检索与读取工具 - 提供案件材料的搜索和验证功能

本模块提供了案件文档的本地检索功能，支持：
1. 全文搜索 - 在案件材料中搜索关键词
2. 页码获取 - 获取指定文档的指定页码内容
3. 原文验证 - 验证引用的原文是否存在于文档中

使用场景：
- 证据分析时检索相关文档
- 文书生成时验证引用内容
- 事实提取时查找关键信息
"""

from __future__ import annotations  # 启用未来类型注解

from typing import Optional  # 可选类型提示

from lexflow_agent.engine.models.case import CaseDocument  # 案件文档模型


class CaseDocReader:
    """案件文档读取工具 - 提供文档搜索、获取和验证功能
    
    该类维护一个文档字典，支持按文档 ID 检索。
    主要用于本地文档的快速搜索和内容验证。
    """

    def __init__(self):
        """初始化文档读取器 - 创建空文档字典"""
        self._documents: dict[str, CaseDocument] = {}  # 文档存储：{document_id: CaseDocument}

    def load_case(self, documents: list[CaseDocument]):
        """加载案件文档 - 将文档列表加载到内存中
        
        Args:
            documents: 案件文档列表
        """
        for doc in documents:
            self._documents[doc.document_id] = doc  # 按文档 ID 存储

    def search(self, case_id: str, query: str, doc_ids: Optional[list[str]] = None) -> list[dict]:
        """搜索案件材料 - 在文档中搜索关键词
        
        Args:
            case_id: 案件 ID（当前未使用，保留用于扩展）
            query: 搜索关键词
            doc_ids: 限定搜索的文档 ID 列表，None 表示搜索所有文档
        
        Returns:
            搜索结果列表，每个结果包含：document_id, page, text, relevance
        """
        results = []
        # 如果指定了 doc_ids，只搜索这些文档；否则搜索所有文档
        docs = [self._documents[did] for did in doc_ids] if doc_ids else self._documents.values()
        q = query.lower()  # 转换为小写（不区分大小写搜索）
        for doc in docs:
            for page in doc.pages:  # 遍历每一页
                if q in page.text.lower():  # 如果页面文本包含关键词
                    results.append({
                        "document_id": doc.document_id,  # 文档 ID
                        "page": page.page,  # 页码
                        "text": page.text[:200],  # 文本片段（最多 200 字符）
                        "relevance": 0.5,  # 相关度评分（固定 0.5，可后续优化）
                    })
        return results[:10]  # 最多返回 10 条结果

    def get_page(self, case_id: str, document_id: str, page: int) -> Optional[str]:
        """获取指定文档的指定页码内容
        
        Args:
            case_id: 案件 ID（当前未使用，保留用于扩展）
            document_id: 文档 ID
            page: 页码
        
        Returns:
            页面文本内容，如果文档或页码不存在则返回 None
        """
        doc = self._documents.get(document_id)  # 获取文档
        if not doc:  # 文档不存在
            return None
        for p in doc.pages:  # 遍历页面
            if p.page == page:  # 找到指定页码
                return p.text  # 返回页面文本
        return None  # 页码不存在

    def verify_quote(self, document_id: str, page: int, quote: str) -> bool:
        """验证原文摘录是否存在 - 检查引用内容是否在指定页面中
        
        Args:
            document_id: 文档 ID
            page: 页码
            quote: 要验证的引用内容
        
        Returns:
            True 表示引用内容存在于页面中，False 表示不存在
        """
        text = self.get_page("", document_id, page)  # 获取页面文本
        if not text:  # 页面不存在
            return False
        return quote in text  # 检查引用是否在文本中

    def clear(self):
        """清空所有文档 - 释放内存"""
        self._documents.clear()


# 全局单例 - 所有节点共享同一个文档读取器
case_doc_reader = CaseDocReader()