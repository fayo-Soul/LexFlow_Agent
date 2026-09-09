# 模块文档字符串：文书生成 Agent
"""文书生成 Agent"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入案件智能体状态
from lexflow_agent.engine.state import CaseAgentState
# 导入文书起草结果模型
from lexflow_agent.engine.models.document import DraftResult
# 导入 LLM 对话调用辅助函数
from lexflow_agent.engine.nodes.llm_helper import llm_chat_call
# 导入 Prompt 管理器
from lexflow_agent.engine.prompts.manager import prompt_manager
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# 支持的文书类型列表
_DOC_TYPES = [
    ("arbitration_application", "劳动仲裁申请书"),  # 仲裁申请书
    ("defense_statement", "劳动争议答辩状"),        # 答辩状
    ("evidence_list", "证据目录"),                  # 证据目录
]


# 文书起草节点函数
@trace_node("draft_documents")
def draft_documents_node(state: CaseAgentState) -> dict:
    # 初始化草稿列表、降级记录和追踪列表
    drafts = list(state.drafts)
    all_degradations = list(state.degradations)
    all_traces = list(state.trace)

    # 遍历每种文书类型进行起草
    for doc_type, _ in _DOC_TYPES:
        # 构建事实文本
        facts = "\n".join(f"- {f.statement}" for f in state.facts) or "(无)"
        # 获取策略描述
        strategy_text = state.strategy.primary_plan.description if state.strategy else "(无)"
        # 加载文书起草 Prompt
        system_prompt = prompt_manager.load("drafting_agent")
        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},  # 系统提示
            {"role": "user", "content": f"文书类型: {dict(_DOC_TYPES).get(doc_type, doc_type)}\n\n确认事实:\n{facts}\n\n已批准策略:\n{strategy_text}"},  # 用户输入
        ]
        # 调用 LLM 生成文书
        text, degraded, degs, trace = llm_chat_call(
            "draft_documents", messages, state, fallback_text=f"【{doc_type} 生成失败】",
        )
        # 收集降级记录和追踪信息
        all_degradations.extend(degs)
        all_traces.append(trace)
        # 添加起草结果
        drafts.append(DraftResult(document_type=doc_type, full_text=text or "", reviewed=False))

    # 返回文书起草结果
    return {"drafts": drafts, "phase": "review", "current_node": "draft_documents",
        "degradations": all_degradations, "trace": all_traces,
        "prompt_versions": {**state.prompt_versions, **prompt_manager.versions}}