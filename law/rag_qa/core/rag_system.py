# 1.导包
# 1.1四种策略模板
from rag_qa.core import vector_store
from rag_qa.core.prompts import RAGPrompt
# 1.2基础模块
import time
from base import setup_logger,Config
# 1.3加载查询分类器与策略选择器
from rag_qa.core.law_query_classifier import LawQueryClassifier
from rag_qa.core.strategy_selector import StrategySelector

logger=setup_logger("rag_system")
# 2.RAGSystem类
class RAGSystem(object):
    def __init__(self,vector_store,llm):
        # 初始化vector_store与llm
        self.vector_store=vector_store
        self.llm=llm
        self.rag_prompt=RAGPrompt.rag_prompt()
        self.query_classifier=LawQueryClassifier()
        # 初始化策略选择器
        self.strategy_selector=StrategySelector()

    # 3.定义一个私有方法_retriever_with_hyde(self,query)
    def _retriever_with_hyde(self,query):
        # 3.1打印输出日志
        logger.info(f"使用HyDE策略进行检索：(查询: {query})")
        # 3.2先获取假设的问题答案 => 调用vector_store的混合检索，检索最佳的前k片段
        hyde_prompt_template=RAGPrompt.hyde_prompt()
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query)).strip()
            logger.info(f"HyDE生成假设答案：{hypo_answer}")
            # 3.3 拿着生成的答案去vectory_store进行混合检索
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer,
                k=Config().RETRIEVAL_K
            )
        except Exception as e:
            logger.error(f"HyDE检索策略执行失败：{e}")
            return []

    # 4. 定义一个私有方法_retrieve_with_subqueries(self, query)
    def _retrieve_with_subqueries(self, query):
        logger.info(f"使用子查询策略进行检索：(查询：{query})")
        subquery_prompt_template = RAGPrompt.subquery_prompt()
        try:
            subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询：{subqueries}")
            if not subqueries:
                logger.warning(f"未能生成有效的子查询")
                return []
            # 定义一个变量all_docs，代表由子查询检索到的所有问题答案
            all_docs = []
            # 遍历所有子查询问题，拿到这个问题然后去向量数据库进行检索操作
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q,
                    k=Config().RETRIEVAL_K
                )
                all_docs.extend(docs)
            # 数据检索完成后，执行去重操作
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())
            logger.info(f"所有子查询检索到的总文档数量：{len(all_docs)}，去重后的文档数据：{len(unique_docs)}")
            return unique_docs

        except Exception as e:
            logger.error(f"子查询检索策略执行失败：{e}")
            return []

    # 5. 定义一个私有方法_retrieve_with_backtracking(self, query)
    def _retrieve_with_backtracking(self, query):
        logger.info(f"使用回溯策略进行检索：(查询：{query})")
        # 获取回溯检索对应的模板
        backtracking_prompt_template = RAGPrompt.backtracking_prompt()
        try:
            simplified_query = self.llm(backtracking_prompt_template.format(query=query)).strip()
            logger.info(f"生成的简化查询：{simplified_query}")
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query,
                k=Config().RETRIEVAL_K
            )
        except Exception as e:
            logger.error(f"回溯检索策略执行失败：{e}")
            return []

    # 6.定义一个公共方法，retrieve_and_merge用于统一查询入口
    def retrieve_and_merge(
        self,
        query,
        source_filter=None,
        strategy=None,
        document_type_filter=None,
        metadata_filters=None,
    ):
        # 获取参数中的strategy，如果为空，则使用策略选择器，获取应该使用哪种策略
        if strategy is None:
            strategy=self.strategy_selector.select_strategy(query) # 直接检索 or 假设答案检索 or 子查询检索 or 回溯问题检索
        # 使用if...elif...else进行策略选择
        ranked_sub_chunks=[]
        if strategy=="回溯问题检索":
            ranked_sub_chunks=self._retrieve_with_backtracking(query)
        elif strategy=="子查询问题检索":
            ranked_sub_chunks=self._retrieve_with_subqueries(query)
        elif strategy=="假设答案检索":
            ranked_sub_chunks=self._retriever_with_hyde(query)
        else:
            # 如果以上三种检索策略都不符合，则使用直接检索
            ranked_sub_chunks=self.vector_store.hybrid_search_with_rerank(
                query,
                k=Config().RETRIEVAL_K,
                source_filter=source_filter,
                document_type_filter=document_type_filter,
                metadata_filters=metadata_filters,
            )
        logger.info(f"策略：{strategy}，共检索到{len(ranked_sub_chunks)}个候选文档")
        final_context_docs=ranked_sub_chunks[:Config().CANDIDATE_M]
        logger.info(f"最终选择：{len(final_context_docs)}个文档作为上下文")
        return final_context_docs

    # 7.generate_answer()生成答案(RAG中的AG，增强+生成)
    def generate_answer(self,query,source_filter=None):
        start_time=time.time() # 记录起始时间
        logger.info(f'开始处理查询：{query}，学科过滤：{source_filter}')

        # 判断查询分类 => 法律问题（返回具体领域）或非法律问题（返回 None）
        query_category=self.query_classifier.predict(query)
        logger.info(f"查询分类结果：{query_category}，查询问题：{query}")

        if query_category is None:
            logger.info(f"非法律问题，直接调用LLM大模型进行处理")
            prompt_input = self.rag_prompt.format(
                context="",
                question=query,
                phone=Config().CUSTOMER_SERVICE_PHONE
            )
            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f"LLM大模型处理失败：{e}")
                answer = f"抱歉，处理您的非法律问题时出错。请联系人工客服：{Config().CUSTOMER_SERVICE_PHONE}"
            process_time = time.time() - start_time
            logger.info(f"非法律问题处理完毕，耗时：{process_time}秒，查询问题：{query}")
            return answer

        # 如果代码执行到这里，则说明是法律问题
        logger.info(f"法律问题，调用RAG模型进行处理")
        # 使用预测的法律领域作为 source_filter
        source_filter = query_category
        # 判断query对应的检索策略
        stretegy=self.strategy_selector.select_strategy(query) # 直接检索 or 假设问题检索 or 子查询检索 or 回溯问题检索
        # 获取检索到的上下文文档(片段)
        context_docs=self.retrieve_and_merge(query,source_filter,stretegy)
        # 对文档进行合并操作 => 变成字符串格式 => 喂给LLM大模型
        if context_docs:
            context="\n\n".join([doc.page_content for doc in context_docs])
            logger.info(f'上下文构建完成，共有{len(context_docs)}个文档')
        else:
            context=""
            logger.info(f'没有检索到任何文档，请使用空上下文获取答案')
        prompt_input=self.rag_prompt.format(
            context=context,
            question=query,
            phone=Config().CUSTOMER_SERVICE_PHONE
        )
        # 调用大模型获取最终答案
        try:
            answer=self.llm(prompt_input)
        except Exception as e:
            logger.error(f"LLM大模型处理失败：{e}")
            answer=f"抱歉，处理您的专业知识问题时出错。请联系人工客服：{Config().CUSTOMER_SERVICE_PHONE}"
        # 获取最终处理时间 与 答案
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
        return answer
