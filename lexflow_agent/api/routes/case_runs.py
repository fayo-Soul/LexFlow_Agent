"""案件分析任务 API 路由 - 提供案件分析的 RESTful API

本模块是 LexFlow Agent 的核心 API 接口，提供以下功能：
1. 创建案件分析任务（POST /case-runs）
2. 查询任务状态（GET /case-runs/{run_id}）
3. 获取任务结果（GET /case-runs/{run_id}/result）
4. 人工审核恢复（POST /case-runs/{run_id}/resume）
5. 任务重试（POST /case-runs/{run_id}/retry）
6. 获取追踪信息（GET /case-runs/{run_id}/trace）
7. 订阅工作流事件（GET /case-runs/{run_id}/events）
8. 提交反馈（POST /case-runs/{run_id}/feedback）
9. 删除任务（DELETE /case-runs/{run_id}）

工作流流程：
案件输入 → 规则过滤 → 意图分类 → 并发控制 → 启动工作流 → 异步执行 → 返回结果
"""

from __future__ import annotations  # 启用未来类型注解

import asyncio  # 异步编程（用于异步执行工作流）
import json  # JSON 序列化（用于 SSE 事件流）
from datetime import datetime  # 时间处理
from typing import Callable, Optional  # 类型提示工具

from fastapi import APIRouter, HTTPException, Header, Depends  # FastAPI 路由和异常
from fastapi.responses import StreamingResponse  # SSE 流响应
from langgraph.types import Command  # LangGraph 命令（用于恢复执行）
from pydantic import BaseModel, Field  # 数据模型验证

from lexflow_agent.config.settings import settings  # 全局配置
from lexflow_agent.engine.models.case import CaseInput, CaseType, ClientRole, CaseDocument  # 案件模型
from lexflow_agent.engine.models.enums import WorkflowStatus  # 工作流状态枚举
from lexflow_agent.engine.models.facts import Fact, FactStatus  # 事实模型
from lexflow_agent.engine.models.response import ApiResponse, ApiError  # API 响应模型
from lexflow_agent.engine.state import CaseAgentState  # 全局状态模型
from lexflow_agent.engine.graph import get_graph, delete_checkpoint  # 工作流图操作
from lexflow_agent.engine.precheck.rule_engine import rule_engine  # 规则过滤引擎
from lexflow_agent.engine.precheck.intent_classifier import intent_classifier  # 意图分类器
from lexflow_agent.engine.runtime import graceful_shutdown, concurrency_controller  # 运行时控制
from lexflow_agent.engine.run_store import run_store  # 运行存储
from lexflow_agent.engine.tracing import get_langsmith_callbacks  # LangSmith 追踪回调

# 从持久化存储加载所有运行记录（API 进程启动时恢复）
_runs: dict[str, CaseAgentState] = run_store.load_all()  # {run_id: CaseAgentState}
# 构建幂等键索引（用于防止重复提交）
_idempotency: dict[str, str] = {
    run.idempotency_key: run_id
    for run_id, run in _runs.items()
    if run.idempotency_key
}


class CreateRunRequest(BaseModel):
    """创建案件分析任务请求模型"""
    case_id: str  # 案件 ID
    case_type: CaseType = CaseType.LABOR_DISPUTE  # 案件类型（默认劳动争议）
    client_role: ClientRole = ClientRole.EMPLOYEE  # 客户角色（默认劳动者）
    client_goal: str = Field(..., min_length=1)  # 客户诉求（必填，至少 1 字符）
    documents: list[CaseDocument] = Field(..., min_length=1)  # 案件文档（必填，至少 1 个）
    idempotency_key: Optional[str] = None  # 幂等键（可选，用于防止重复提交）


class ResumeRequest(BaseModel):
    """恢复执行请求模型（人工审核后继续）"""
    action: str  # 审核动作（approve_facts, modify_facts, approve_strategy, submit_clarification 等）
    comments: str = ""  # 审核意见
    modified_facts: list[dict] = Field(default_factory=list)  # 修改后的事实列表
    selected_strategy: str = ""  # 选择的策略
    documents: list[CaseDocument] = Field(default_factory=list)  # 补充的文档
    clarification_answers: dict[str, str] = Field(default_factory=dict)  # ★新增：澄清回答


class FeedbackRequest(BaseModel):
    """反馈请求模型（用于 LangSmith 评估）"""
    langsmith_run_id: str  # LangSmith 运行 ID
    key: str = "user_score"  # 反馈键（默认 user_score）
    score: float = Field(..., ge=0, le=1)  # 评分（0-1 之间）
    comment: str = ""  # 反馈评论


def _to_case_input(req: CreateRunRequest) -> CaseInput:
    """将创建请求转换为案件输入模型
    
    Args:
        req: 创建运行请求对象
    
    Returns:
        CaseInput 对象，用于后续处理
    """
    return CaseInput(
        case_id=req.case_id, case_type=req.case_type,
        client_role=req.client_role, client_goal=req.client_goal,
        documents=req.documents, idempotency_key=req.idempotency_key,
    )


def _check_auth(authorization: Optional[str] = Header(None)):
    """认证检查 - 验证 API Token
    
    Args:
        authorization: Authorization 请求头（Bearer Token 格式）
    
    Raises:
        HTTPException: 认证失败时抛出 401 错误
    """
    token = (authorization or "").removeprefix("Bearer ").strip()  # 提取 Token
    if not token or token != settings.API_TOKEN:  # Token 为空或不匹配
        raise HTTPException(status_code=401, detail="Unauthorized")


router = APIRouter(dependencies=[Depends(_check_auth)])  # 创建路由器，所有路由自动应用认证检查


def _save_run(run: CaseAgentState) -> None:
    """保存运行状态到持久化存储
    
    更新 updated_at 时间戳并保存到 run_store（SQLite 数据库）。
    
    Args:
        run: 案件运行状态对象
    """
    run.updated_at = datetime.now()  # 更新最后修改时间
    run_store.save(run)  # 保存到数据库


def _inject_langsmith_tracer(config: dict) -> None:
    """向 LangGraph config 注入 LangChainTracer callback。

    仅在 LANGSMITH_TRACING 开启且 API Key 已配置时生效。
    修改 config dict in-place，追加 callbacks 列表。
    """
    callbacks = get_langsmith_callbacks()
    if callbacks:
        existing = config.setdefault("callbacks", [])
        existing.extend(callbacks)


def _stream_graph(graph, graph_input, config, run: CaseAgentState) -> dict:
    """执行 LangGraph 工作流并发布节点级状态更新（用于 SSE）
    
    该函数同步执行 LangGraph 的 stream 方法，在每次节点更新时：
    1. 更新 run 状态的对应字段
    2. 记录工作流事件
    3. 保存到持久化存储
    
    Args:
        graph: LangGraph 编译后的应用对象
        graph_input: 图输入（初始状态或 None 表示从检查点恢复）
        config: LangGraph 配置（包含 thread_id、tags、metadata）
        run: 案件运行状态对象
    
    Returns:
        最后一次节点更新的字典
    """
    latest = {}  # 记录最新的节点更新
    for namespace, update in graph.stream(
        graph_input,
        config,
        stream_mode="updates",  # 流模式：updates（节点更新）
        subgraphs=True,  # 包含子图更新
    ):
        if not isinstance(update, dict):
            continue
        for node, values in update.items():
            if not isinstance(values, dict):
                continue
            latest.update(values)  # 更新最新值
            for key, value in values.items():
                if hasattr(run, key):  # 如果 run 对象有该字段
                    setattr(run, key, value)  # 更新字段值
            event = {
                "sequence": len(run.workflow_events) + 1,  # 事件序列号
                "time": datetime.now().isoformat(),  # 事件时间
                "agent": run.current_agent,  # 当前 Agent
                "node": node,  # 节点名称
                "namespace": list(namespace),  # 命名空间
                "phase": run.phase,  # 当前阶段
                "status": run.status.value if hasattr(run.status, "value") else str(run.status),  # 状态值
                "degraded": bool(run.degradations),  # 是否降级
            }
            run.workflow_events.append(event)  # 记录事件
            _save_run(run)  # 保存到数据库
    return latest


@router.post("/case-runs", status_code=201)
async def create_case_run(req: CreateRunRequest, authorization: Optional[str] = Header(None)):
    """创建案件分析任务并启动工作流
    
    处理流程：
    1. 认证检查
    2. 幂等检查（防止重复提交）
    3. 规则过滤（校验输入合法性）
    4. 意图分类（判断是否为案件分析请求）
    5. 并发控制（同一律师的任务限制）
    6. 创建初始状态
    7. 异步启动工作流
    8. 返回 run_id
    
    Args:
        req: 创建运行请求
        authorization: 认证头
    
    Returns:
        ApiResponse 包含 run_id 和初始状态
    """
    _check_auth(authorization)  # 认证检查

    # 幂等检查 - 如果已有相同幂等键的运行，直接返回
    if req.idempotency_key:
        existing_run_id = _idempotency.get(req.idempotency_key)
        if existing_run_id and existing_run_id in _runs:
            run = _runs[existing_run_id]
            return ApiResponse(
                status="success",
                data={"run_id": existing_run_id, "status": run.status.value},
            )

    # 规则过滤 - 校验输入合法性（案由、材料、诉求等）
    case_input = _to_case_input(req)
    rule_result = rule_engine.check(case_input)
    if rule_result:  # 规则过滤失败
        return ApiResponse(
            status="error",
            error=ApiError(code=rule_result.code, message=rule_result.message, details={"log_audit": rule_result.log_audit}),
        )

    # 意图分类 - 判断请求类型
    intent_result = intent_classifier.classify(case_input)
    if intent_result.intent.value != "case_analysis":  # 非案件分析请求
        return ApiResponse(
            status="success",
            data={"intent": intent_result.intent.value, "confidence": intent_result.confidence},
            warnings=[f"请求被识别为 {intent_result.intent.value}，非案件分析请求"],
        )

    # 并发控制 - 限制同一律师的并发任务数
    lawyer_id = f"lawyer_{req.case_id}"
    can_run = await concurrency_controller.acquire(lawyer_id)
    if not can_run:  # 并发限制
        return ApiResponse(
            status="error",
            error=ApiError(code="CONCURRENCY_LIMIT", message="同一律师的上一任务仍在进行中，请稍后再试"),
        )

    # 创建初始状态
    run_id = f"run_{req.case_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    initial_state = CaseAgentState(
        run_id=run_id, case_id=req.case_id,
        idempotency_key=req.idempotency_key,
        case_input=case_input, status=WorkflowStatus.PENDING,
    )
    _runs[run_id] = initial_state  # 保存到内存
    _save_run(initial_state)  # 保存到数据库
    if req.idempotency_key:
        _idempotency[req.idempotency_key] = run_id  # 更新幂等键索引

    # 异步启动工作流（不阻塞 API 响应）
    import asyncio
    asyncio.create_task(_run_workflow(run_id, result_callback=_release_lawyer(lawyer_id)))

    return ApiResponse(status="success", data={"run_id": run_id, "status": "pending"})


def _release_lawyer(lawyer_id: str):
    """创建律师并发锁释放回调函数
    
    Args:
        lawyer_id: 律师 ID
    
    Returns:
        回调函数，执行时释放律师并发锁
    """
    def callback():
        concurrency_controller.release(lawyer_id)  # 释放并发锁
    return callback


def _apply_graph_result(run: CaseAgentState, result: dict, next_nodes: tuple[str, ...]) -> None:
    """应用 LangGraph 执行结果到运行状态
    
    根据图执行结果和下一步节点，更新运行状态：
    - 如果下一步是 legal_research 或 identify_issues → 等待事实复核
    - 如果下一步是 document_generation 或 draft_documents → 等待策略复核
    - 如果状态是等待人工干预 → 保持不变
    - 否则 → 标记为完成
    
    Args:
        run: 案件运行状态对象
        result: 图执行结果字典
        next_nodes: 下一步节点元组
    """
    for key, value in result.items():
        setattr(run, key, value)  # 更新状态字段

    if {"legal_research_agent", "identify_issues"} & set(next_nodes):  # 需要事实复核
        run.status = WorkflowStatus.WAITING_FACT_REVIEW
    elif {"document_generation_agent", "draft_documents"} & set(next_nodes):  # 需要策略复核
        run.status = WorkflowStatus.WAITING_STRATEGY_REVIEW
    elif run.status == WorkflowStatus.WAITING_HUMAN_INTERVENTION:  # 人工干预状态保持不变
        pass
    else:  # 其他情况标记为完成
        run.status = WorkflowStatus.COMPLETED
        run.completed_at = datetime.now()


async def _run_workflow(run_id: str, result_callback=None):
    """异步执行工作流 - 在后台线程中运行 LangGraph
    
    该函数在 asyncio 任务中执行，不阻塞 API 响应。
    执行流程：
    1. 获取运行状态
    2. 获取 LangGraph 实例
    3. 设置状态为 RUNNING
    4. 构建配置（thread_id, tags, metadata）
    5. 检查是否有现有检查点（断点续传）
    6. 执行工作流（带超时控制）
    7. 应用执行结果
    8. 异常处理（标记为 FAILED）
    9. 释放律师并发锁（回调）
    
    Args:
        run_id: 运行 ID
        result_callback: 结果回调函数（通常用于释放并发锁）
    """
    run = _runs.get(run_id)  # 获取运行状态
    if not run:  # 运行不存在
        return

    try:
        graph = get_graph()  # 获取 LangGraph 实例
        run.status = WorkflowStatus.RUNNING  # 设置状态为运行中
        _save_run(run)  # 保存状态

        # 构建 LangGraph 配置
        config = {
            "configurable": {"thread_id": run_id},  # 线程 ID（用于检查点）
            "tags": ["lexflow", "four-agent", "labor-dispute"],  # 标签（用于追踪）
            "metadata": {
                "thread_id": run_id,  # 线程 ID
                "session_id": run_id,  # 会话 ID
                "case_id": run.case_id,  # 案件 ID
                "deployment": "vm-192.168.88.100",  # 部署信息
                "app_version": settings.APP_VERSION,  # 应用版本
            },
        }
        # 注入 LangSmith tracer callback — 使 LangSmith 面板可追踪完整调用链
        _inject_langsmith_tracer(config)
        # 检查是否有现有检查点（断点续传）
        existing = await asyncio.to_thread(graph.get_state, config)
        graph_input = None if existing.values else run.model_dump()  # 无检查点则传入初始状态
        
        # 执行工作流（带超时控制，在后台线程中运行同步代码）
        result = await asyncio.wait_for(
            asyncio.to_thread(_stream_graph, graph, graph_input, config, run),
            timeout=settings.WORKFLOW_TIMEOUT_SECONDS,  # 工作流超时
        )
        if isinstance(result, dict):  # 如果有执行结果
            snapshot = await asyncio.to_thread(graph.get_state, config)  # 获取最新状态
            _apply_graph_result(run, dict(snapshot.values), tuple(snapshot.next))  # 应用结果
            _save_run(run)  # 保存状态
    except Exception as e:  # 捕获所有异常
        run.status = WorkflowStatus.FAILED  # 标记为失败
        run.errors.append(str(e))  # 记录错误信息
        _save_run(run)  # 保存状态
    finally:
        if result_callback:  # 执行回调（释放并发锁）
            result_callback()


@router.get("/case-runs/{run_id}")
async def get_run_status(run_id: str):
    """查询任务状态 - 返回运行的基本信息
    
    Args:
        run_id: 运行 ID
    
    Returns:
        ApiResponse 包含运行状态、当前 Agent、阶段等信息
    
    Raises:
        HTTPException: 404 如果运行不存在
    """
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="运行不存在")
    return ApiResponse(status="success", data={
        "run_id": run.run_id,  # 运行 ID
        "status": run.status.value,  # 状态值
        "current_agent": run.current_agent,  # 当前 Agent
        "current_node": run.current_node,  # 当前节点
        "phase": run.phase,  # 当前阶段
        "revision_count": run.revision_count,  # 修订次数
        "errors": run.errors[:5],  # 最近 5 个错误
        "warnings": run.warnings[:5],  # 最近 5 个警告
    })


@router.get("/case-runs/{run_id}/result")
async def get_run_result(run_id: str):
    """获取任务完整结果 - 返回所有产出数据
    
    返回完整的输出包，包含：facts, issues, evidence, strategy, drafts, review 等。
    
    Args:
        run_id: 运行 ID
    
    Returns:
        ApiResponse 包含完整的案件分析结果
    
    Raises:
        HTTPException: 404 如果运行不存在
    """
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="运行不存在")
    return ApiResponse(status="success", data=run.to_output_package())  # 返回完整输出包


@router.post("/case-runs/{run_id}/resume")
async def resume_run(run_id: str, req: ResumeRequest):
    """恢复人工审核后继续执行
    
    支持的操作：
    - approve_facts: 批准事实，继续执行
    - modify_facts: 修改事实，重新理解
    - supplement_materials: 补充材料，重新理解
    - approve_strategy: 批准策略，继续执行
    - modify_strategy: 修改策略，重新生成
    - supplement_research: 补充研究，重新研究
    - cancel: 取消任务
    
    Args:
        run_id: 运行 ID
        req: 恢复请求（包含动作、评论、修改等）
    
    Returns:
        ApiResponse 包含恢复后的状态
    
    Raises:
        HTTPException: 404 如果运行不存在，409 如果当前状态不允许该操作
    """
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="运行不存在")

    if req.action == "cancel":  # 取消任务
        run.status = WorkflowStatus.CANCELLED
        _save_run(run)
        concurrency_controller.release(f"lawyer_{run.case_id}")  # 释放并发锁
        return ApiResponse(status="success", data={"status": "cancelled"})

    # 定义各状态允许的操作
    allowed_actions = {
        WorkflowStatus.WAITING_FACT_REVIEW: {
            "approve_facts", "modify_facts", "supplement_materials",
        },
        WorkflowStatus.WAITING_STRATEGY_REVIEW: {
            "approve_strategy", "modify_strategy", "supplement_research",
        },
        WorkflowStatus.WAITING_CLARIFICATION: {  # ★新增
            "submit_clarification",
        },
    }
    if req.action not in allowed_actions.get(run.status, set()):  # 操作不允许
        raise HTTPException(status_code=409, detail="当前工作流状态不允许该审核动作")

    run.warnings.append(f"人工审核: {req.action}; {req.comments}".rstrip("; "))  # 记录审核意见
    try:
        graph = get_graph()
        config = {
            "configurable": {"thread_id": run_id},
            "tags": ["lexflow", "human-review"],
            "metadata": {
                "thread_id": run_id,
                "session_id": run_id,
                "case_id": run.case_id,
                "review_action": req.action,
                "deployment": "vm-192.168.88.100",
            },
        }
        graph_input = None

        # 如果不是直接批准，需要构建更新和路由
        if req.action not in {"approve_facts", "approve_strategy"}:
            updates: dict = {"warnings": run.warnings}
            # 根据操作决定路由目标
            target_map = {
                "modify_facts": "case_understanding_agent",  # 修改事实 → 重新理解
                "supplement_materials": "case_understanding_agent",  # 补充材料 → 重新理解
                "modify_strategy": "case_strategy_agent",  # 修改策略 → 重新生成策略
                "supplement_research": "legal_research_agent",  # 补充研究 → 重新研究
            }

            if req.action == "submit_clarification":  # ★新增
                if req.clarification_answers:
                    # 把回答写入 context（后续节点会读取）
                    updates["clarification_answers"] = req.clarification_answers
                    # 回答注入到 facts 文本（供 document_generation 使用）
                    from lexflow_agent.engine.models.facts import Fact
                    combined_answers = "\n".join(
                        f"{q_id}: {answer}"
                        for q_id, answer in req.clarification_answers.items()
                    )
                    updates["facts"] = run.facts + [
                        Fact(statement=f"用户澄清: {combined_answers}",
                             status=FactStatus.CONFIRMED,
                             source="clarification_answers")
                    ]
                # 路由到文档生成
                target = "document_generation_agent"
            else:
                target = target_map.get(req.action, "case_understanding_agent")
            if req.modified_facts:  # 如果有修改的事实
                from lexflow_agent.engine.models.facts import Fact
                updates["facts"] = [Fact.model_validate(item) for item in req.modified_facts]
            if req.documents and run.case_input:  # 如果有补充文档
                documents = {doc.document_id: doc for doc in run.case_input.documents}
                documents.update({doc.document_id: doc for doc in req.documents})
                updates["case_input"] = run.case_input.model_copy(
                    update={"documents": list(documents.values())}
                )
            if req.selected_strategy:  # 如果有律师选择的策略
                updates["warnings"] = run.warnings + [f"律师策略意见: {req.selected_strategy}"]
            graph_input = Command(update=updates, goto=target)  # 构建 LangGraph 命令

        run.status = WorkflowStatus.RUNNING  # 设置状态为运行中
        result = await asyncio.wait_for(
            asyncio.to_thread(_stream_graph, graph, graph_input, config, run),
            timeout=settings.WORKFLOW_TIMEOUT_SECONDS,
        )
        if isinstance(result, dict):
            snapshot = await asyncio.to_thread(graph.get_state, config)
            _apply_graph_result(run, dict(snapshot.values), tuple(snapshot.next))
            _save_run(run)
    except Exception as e:
        run.status = WorkflowStatus.FAILED
        run.errors.append(str(e))
        _save_run(run)

    return ApiResponse(status="success", data={"status": run.status.value})


@router.post("/case-runs/{run_id}/retry")
async def retry_node(run_id: str):
    """任务重试 - 重新执行失败或等待人工干预的任务
    
    适用状态：
    - FAILED: 任务失败
    - WAITING_HUMAN_INTERVENTION: 等待人工干预
    
    处理流程：
    1. 检查运行是否存在
    2. 检查状态是否允许重试
    3. 获取并发锁
    4. 增加重试计数
    5. 设置状态为 PENDING
    6. 异步启动工作流
    
    Args:
        run_id: 运行 ID
    
    Returns:
        ApiResponse 包含新的状态和重试次数
    
    Raises:
        HTTPException: 404 如果运行不存在，409 如果状态不允许重试
    """
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="运行不存在")
    if run.status not in {WorkflowStatus.FAILED, WorkflowStatus.WAITING_HUMAN_INTERVENTION}:
        raise HTTPException(status_code=409, detail="当前状态不允许重试")
    lawyer_id = f"lawyer_{run.case_id}"
    if not await concurrency_controller.acquire(lawyer_id):  # 获取并发锁
        return ApiResponse(
            status="error",
            error=ApiError(code="CONCURRENCY_LIMIT", message="该案件已有任务正在执行"),
        )
    run.retry_count += 1  # 增加重试计数
    run.status = WorkflowStatus.PENDING  # 重置状态为待处理
    _save_run(run)
    asyncio.create_task(_run_workflow(run_id, result_callback=_release_lawyer(lawyer_id)))  # 异步启动
    return ApiResponse(status="success", data={"status": "pending", "retry_count": run.retry_count})


async def recover_incomplete_runs() -> None:
    """恢复未完成的任务 - 在 API 进程重启时恢复运行中的任务
    
    遍历所有运行记录，如果状态为 PENDING 或 RUNNING（表示上次进程停止时仍在执行），
    则重新启动这些任务。
    """
    for run in list(_runs.values()):  # 遍历所有运行记录
        if run.status not in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING}:  # 只恢复运行中的任务
            continue
        lawyer_id = f"lawyer_{run.case_id}"
        if await concurrency_controller.acquire(lawyer_id):  # 获取并发锁
            asyncio.create_task(
                _run_workflow(run.run_id, result_callback=_release_lawyer(lawyer_id))
            )


@router.get("/case-runs/{run_id}/trace")
async def get_trace(run_id: str):
    """获取任务追踪信息 - 返回节点执行轨迹
    
    Args:
        run_id: 运行 ID
    
    Returns:
        ApiResponse 包含所有节点的执行轨迹（时间、token 消耗、状态等）
    
    Raises:
        HTTPException: 404 如果运行不存在
    """
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="运行不存在")
    return ApiResponse(status="success", data={
        "traces": [t.model_dump() for t in run.trace],  # 序列化所有追踪记录
    })


@router.get("/case-runs/{run_id}/events")
async def stream_run_events(run_id: str):
    """订阅工作流事件 - 使用 Server-Sent Events (SSE) 流式推送状态变更
    
    该接口提供实时的事件推送服务，客户端可以通过 SSE 接收：
    1. node 事件：每次节点执行时推送
    2. workflow 事件：工作流状态变化时推送
    
    当事件到达终态（COMPLETED, FAILED, CANCELLED, 或等待人工审核）时，流自动关闭。
    
    Args:
        run_id: 运行 ID
    
    Returns:
        StreamingResponse SSE 流（text/event-stream 格式）
    
    Raises:
        HTTPException: 404 如果运行不存在
    """
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="运行不存在")

    async def events():
        """SSE 事件生成器"""
        cursor = 0  # 事件游标（记录已推送的事件数量）
        previous_state = None  # 上一次的工作流状态（用于去重）
        terminal = {  # 终态集合（到达这些状态时停止推送）
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
            WorkflowStatus.WAITING_FACT_REVIEW,
            WorkflowStatus.WAITING_STRATEGY_REVIEW,
            WorkflowStatus.WAITING_CLARIFICATION,
            WorkflowStatus.WAITING_HUMAN_INTERVENTION,
        }
        while True:
            run = _runs.get(run_id)
            if run is None:  # 运行被删除
                return
            # 推送所有未推送的新事件
            while cursor < len(run.workflow_events):
                payload = run.workflow_events[cursor]
                cursor += 1
                yield f"event: node\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
            # 推送工作流状态（仅在状态变化时）
            payload = {
                "run_id": run.run_id,
                "status": run.status.value,
                "agent": run.current_agent,
                "node": run.current_node,
                "phase": run.phase,
                "revision_count": run.revision_count,
                "degraded": bool(run.degradations),
            }
            serialized = json.dumps(payload, ensure_ascii=False, default=str)
            if serialized != previous_state:  # 状态发生变化
                yield f"event: workflow\ndata: {serialized}\n\n"
                previous_state = serialized
            if run.status in terminal:  # 到达终态，停止推送
                return
            await asyncio.sleep(0.5)  # 等待 0.5 秒后继续检查

    return StreamingResponse(events(), media_type="text/event-stream")  # 返回 SSE 流


@router.post("/case-runs/{run_id}/feedback")
async def submit_feedback(run_id: str, req: FeedbackRequest):
    """Attach online feedback to a LangSmith run and queue low scores."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="运行不存在")
    from langsmith import Client

    client = Client(
        api_key=settings.LANGSMITH_API_KEY,
        api_url=settings.LANGSMITH_ENDPOINT,
    )
    feedback = await asyncio.to_thread(
        client.create_feedback,
        req.langsmith_run_id,
        req.key,
        score=req.score,
        comment=req.comment,
        source_info={"api_run_id": run_id},
    )
    queued = False
    if req.score < 0.6:
        queues = await asyncio.to_thread(
            lambda: list(client.list_annotation_queues(name="lexflow-production-review"))
        )
        if queues:
            await asyncio.to_thread(
                client.add_runs_to_annotation_queue,
                queues[0].id,
                run_ids=[req.langsmith_run_id],
            )
            queued = True
    return ApiResponse(status="success", data={
        "feedback_id": str(feedback.id),
        "queued_for_annotation": queued,
    })


@router.delete("/case-runs/{run_id}", status_code=204)
async def delete_run(run_id: str):
    run = _runs.pop(run_id, None)
    if run and run.idempotency_key:
        _idempotency.pop(run.idempotency_key, None)
    run_store.delete(run_id)
    await asyncio.to_thread(delete_checkpoint, run_id)