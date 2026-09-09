# 1. 导包
# 导入 LangChain 提示模板
from langchain_core.prompts import PromptTemplate
# 导入日志和配置
from base import setup_logger, Config
# 导入 OpenAI
from openai import OpenAI

logger = setup_logger("strategy_selector")

# 2. 封装一个类，用于策略选择
class StrategySelector(object):
    def __init__(self):
        # 2.1 创建client客户端，本质就是OpenAI类对象
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
            api_key=Config().DASHSCOPE_API_KEY,
            base_url=Config().DASHSCOPE_BASE_URL,
        )
        # 2.2 定义一个属性，用于选择策略模板
        self.strategy_prompt_template = self._get_strategy_prompt()
    # 3. 定义一个call_dashscope方法，用于调用DashScope API
    def call_dashscope(self, prompt):
        # 基于self.client向DashScope API发起请求
        try:
            completion = self.client.chat.completions.create(
                model=Config().LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"},
                    {"role": "user", "content": prompt}
                ],
                # 偏向于专业度的结果
                temperature=0.1
            )
            return completion.choices[0].message.content if completion.choices else "直接检索"
        except Exception as e:
            logger.error(f"DashScope API调用失败：{e}")
            return "直接检索"

    # 4. 定义一个选择策略方法，用于设置发给大模型的Prompt模板
    def _get_strategy_prompt(self):
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：AI 学科学费是多少？
                        * 策略：直接检索
                    * 查询：JAVA的课程大纲是什么？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：人工智能在教育领域的应用有哪些？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )

    # 5.定义一个select_strategy方法，用于选择策略
    def select_strategy(self, query):
        # 5.1 调用DashScope API，获取策略名称
        strategy=self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        # 5.2打印日志并返回策略名称
        logger.info(f"策略选择结果：{strategy}")
        return strategy

if __name__ == '__main__':
    ss=StrategySelector()
    ss.select_strategy("人工智能在教育领域的应用有哪些？")