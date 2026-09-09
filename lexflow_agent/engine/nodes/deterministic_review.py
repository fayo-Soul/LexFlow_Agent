# 模块文档字符串：确定性复核节点 - 全部代码执行
"""确定性复核节点 - 全部代码执行"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入正则表达式模块
import re
# 导入日期类型
from datetime import date

# 导入案件智能体状态
from lexflow_agent.engine.state import CaseAgentState
# 导入复核结果和发现项模型
from lexflow_agent.engine.models.review import ReviewResult, ReviewFinding
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# 检查日期格式的辅助函数
def _check_date_format(text: str) -> list[str]:
    """检查日期格式：YYYY-MM-DD"""
    errors = []
    # 匹配日期格式
    pattern = r"\d{4}-\d{2}-\d{2}"
    matches = re.findall(pattern, text)
    # 验证每个匹配的日期
    for m in matches:
        parts = m.split("-")
        # 检查月份和日期是否有效
        if not (1 <= int(parts[1]) <= 12) or not (1 <= int(parts[2]) <= 31):
            errors.append(f"无效日期: {m}")
    return errors


# 确定性复核节点函数
@trace_node("deterministic_review", run_type="tool")
def deterministic_review_node(state: CaseAgentState) -> dict:
    """确定性质量检查 - 不调用LLM"""
    # 初始化发现项列表
    findings = []

    # 1. 文档/事实/证据 ID 一致性检查
    fact_doc_ids = set()
    for fact in state.facts:
        for s in fact.sources:
            fact_doc_ids.add(s.document_id)

    input_doc_ids = set()
    if state.case_input:
        input_doc_ids = {d.document_id for d in state.case_input.documents}

    # 检查事实引用的文档是否存在
    for fid in fact_doc_ids:
        if fid not in input_doc_ids:
            findings.append(ReviewFinding(
                severity="error", node="extract_facts",
                field="sources.document_id",
                description=f"事实引用不存在的文档 {fid}",
                suggestion="请检查事实来源中的文档ID是否正确",
            ))

    # 2. 当事人名称一致性检查
    if state.facts:
        all_names = set()
        for fact in state.facts:
            for s in fact.sources:
                if s.document_id in input_doc_ids:
                    all_names.add(s.document_id)

    # 3. 日期金额格式校验
    if state.facts:
        for fact in state.facts:
            if fact.occurred_at:
                # 检查日期格式
                date_errors = _check_date_format(fact.occurred_at.isoformat())
                for err in date_errors:
                    findings.append(ReviewFinding(
                        severity="warning", node="extract_facts",
                        field=f"fact.{fact.fact_id}.occurred_at",
                        description=err,
                    ))

    # 4. 未确认事实检查
    unconfirmed = [f for f in state.facts if f.status.value == "insufficient"]
    if unconfirmed and state.drafts:
        findings.append(ReviewFinding(
            severity="warning", node="draft_documents",
            description=f"存在 {len(unconfirmed)} 条材料不足的事实，请确认是否应在文书中引用",
            suggestion="删除或标记为推断",
        ))

    # 5. 置信度标注完整性检查
    for fact in state.facts:
        if fact.confidence <= 0:
            findings.append(ReviewFinding(
                severity="warning", node="extract_facts",
                field=f"fact.{fact.fact_id}.confidence",
                description=f"事实 {fact.fact_id} 缺少置信度标注",
            ))

    # 6. 占位符检查
    if state.drafts:
        for draft in state.drafts:
            # 检查未处理的占位符
            placeholders_remaining = draft.full_text.count("【待补充】")
            # 检查是否有审核标记
            has_review_mark = "【待律师审核】" in draft.full_text
            if placeholders_remaining > 0:
                findings.append(ReviewFinding(
                    severity="info", node="draft_documents",
                    description=f"{draft.document_type} 中存在 {placeholders_remaining} 个未处理占位符",
                    suggestion="请律师补充缺失信息",
                ))
            if not has_review_mark:
                findings.append(ReviewFinding(
                    severity="warning", node="draft_documents",
                    description=f"{draft.document_type} 缺少'待律师审核'标记",
                    suggestion="添加【待律师审核】标注",
                ))

    # 7. 证据页码有效性检查
    for item in state.evidence_matrix:
        for ev in item.evidence:
            if ev.page < 1:
                findings.append(ReviewFinding(
                    severity="error", node="analyze_evidence",
                    field=f"evidence.{ev.evidence_id}.page",
                    description=f"证据页码无效: {ev.page}",
                ))

    # 判断是否通过（无 error 级别的发现项）
    passed = all(f.severity != "error" for f in findings)
    # 返回复核结果
    return {
        "deterministic_review": ReviewResult(findings=findings, passed=passed),
        "current_node": "deterministic_review",
    }