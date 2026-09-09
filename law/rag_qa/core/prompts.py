# 1.导包
from langchain_core.prompts import PromptTemplate

# 2.定义一个RAGPrompt类 => 存在4种检索策略
# 直接检索：问题比较明确，则直接检索
# Hyde检索：先假设一个问题答案，然后进行向量检索，再进行重排序，最后进行NLI判断
# 子查询检索：把用户问题，拆分成多个子问题，然后进行向量检索，再进行重排序，最后进行NLI判断
# 回溯检索：把用户问题调整为更基础、更容易检索的问题，再逐步返回原问题
# query => RAGFlow/Dify/Coze知识库，查询改写 => 策略更多一些
class RAGPrompt(object):
    # 直接检索`
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
            你是一个智能助手，帮助用户回答问题。
                如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
                如果答案来源于检索到的文档，请在回答中说明。

                上下文: {context}
                问题: {question}

                如果无法回答，请回复："信息不足，无法回答，请联系人工客服，电话：{phone}。"
                回答:
            """,
            input_variables=["context", "question", "phone"],
        )

    @staticmethod
    def law_rag_prompt():
        """律所法律研究场景的证据约束生成提示。"""
        return PromptTemplate(
            template="""
            你是律所内部使用的法律研究辅助系统。你的任务是整理检索证据，不是代替律师出具正式法律意见。

            严格规则：
            1. 只能使用“检索证据”中的内容，不得使用证据之外的法条、案号、法院、日期或结论。
            2. 每项法律规则、司法解释观点和案例观点后必须标注证据已有的引用编号，例如 [法规1]、[司法解释1]、[案例1]。
            3. 不得创造、修改或猜测引用编号。
            4. 区分资料原文与初步分析，不得承诺案件结果或胜诉概率。
            5. 事实不足时列出需要进一步核实的事实。
            6. 证据不足以支持结论时，明确写“根据当前知识库资料无法确定”，不得自行补充。

            请按以下结构输出；没有对应证据的部分写“当前知识库未检索到相关资料”：
            一、问题概述
            二、相关法律规定
            三、相关司法解释
            四、相关案例及裁判观点
            五、初步分析
            六、需要进一步核实的事实

            【检索证据】
            {context}

            【研究问题】
            {question}

            【研究辅助回答】
            """,
            input_variables=["context", "question", "phone"],
        )

    # Hyde检索
    @staticmethod
    def hyde_prompt():
        return PromptTemplate(
            template='''
             假设你是用户，想了解以下问题，请生成一个简短的假设答案：
            问题: {query}
            假设答案:
            ''',
            input_variables=["query"],
        )
    # 子查询检索
    @staticmethod
    def subquery_prompt():
        return PromptTemplate(
            template='''
            将以下复杂查询分解为多个简单子查询，每行一个子查询：
            查询: {query}
            子查询:
            ''',
            input_variables=["query"],
        )
    # 回溯检索
    @staticmethod
    def backtracking_prompt():
        return PromptTemplate(
            template='''
            将以下复杂查询简化为一个更简单的问题：
            查询: {query}
            简化问题:
            ''',
            input_variables=["query"],
        )
