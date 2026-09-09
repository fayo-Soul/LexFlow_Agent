"""检索前的轻量意图路由。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRoute:
    intent: str
    answer: str = ""


PRODUCT_PHRASES = (
    "能帮我查询什么",
    "能帮我干什么",
    "可以帮我干什么",
    "能帮我做什么",
    "可以帮我做什么",
    "可以查询什么",
    "能做什么",
    "都会什么",
    "会做什么",
    "有什么功能",
    "怎么使用",
    "如何使用",
    "系统介绍",
)
GREETINGS = {"你好", "您好", "嗨", "hello", "hi", "在吗"}
COURTESY = {"谢谢", "感谢", "好的", "明白了", "再见"}
OUT_OF_SCOPE_PHRASES = (
    "天气",
    "菜谱",
    "写代码",
    "翻译英语",
    "电影推荐",
    "旅游攻略",
    "股票价格",
    "体育比分",
)


def route_query(query: str) -> QueryRoute:
    normalized = "".join(str(query).strip().lower().split()).rstrip("，。！？?!")
    if any(phrase in normalized for phrase in PRODUCT_PHRASES):
        return QueryRoute(
            "product_help",
            "我是律所法律知识库研究助手，可以根据自然语言问题检索法律法规、司法解释和已发布的律所内部资料，"
            "并生成带法律名称、具体条款和原文引用的辅助回答。"
            "当前已重点评测婚姻家事、劳动争议、合同责任、民事诉讼时效和无证驾驶处罚。"
            "你可以直接描述案件事实、争议焦点，或者输入法条名称和条号；超出当前知识库范围时，我会明确提示证据不足。",
        )
    if normalized in GREETINGS:
        return QueryRoute(
            "greeting",
            "你好，我是律所法律知识库 RAG 系统。请告诉我需要研究的法律问题、争议焦点或案件事实。",
        )
    if normalized in COURTESY:
        return QueryRoute("courtesy", "不客气。需要继续检索法规、司法解释或案例时，可以直接提出问题。")
    if any(phrase in normalized for phrase in OUT_OF_SCOPE_PHRASES):
        return QueryRoute(
            "out_of_scope",
            "该问题不属于本系统的法律研究范围。本系统主要提供法规、司法解释、案例和律所内部资料检索。",
        )
    return QueryRoute("legal_research")
