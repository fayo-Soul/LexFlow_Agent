const $ = (id) => document.getElementById(id);
const state = { runId: null, status: null, result: null, events: [], tab: "facts", timer: null, demoDocuments: null };
const statusLabels = {pending:"等待启动",running:"处理中",waiting_fact_review:"等待事实审核",waiting_strategy_review:"等待策略审核",waiting_human_intervention:"需要人工处理",completed:"分析完成",failed:"执行失败",cancelled:"已取消"};
const agentLabels = {case_understanding_agent:"案件理解 Agent",legal_research_agent:"法律研究 Agent",case_strategy_agent:"案件策略 Agent",document_generation_agent:"文书生成 Agent"};
const demoDocuments = [
  {document_id:"doc_001_001",file_name:"劳动合同.pdf",pages:[
    {page:1,text:"甲方：广州科技有限公司，统一社会信用代码 91440101MA5ABCDEF\n乙方：测试员工01，身份证号 440106199001011234\n\n劳动合同期限：自2022年1月1日起至2025年12月31日止。\n工作岗位：高级软件工程师\n工作地点：广州市天河区\n月工资：15,000元（税前），每月10日发放上月工资。"},
    {page:2,text:"第十二条 解除和终止\n……\n用人单位解除劳动合同的，应当提前三十日书面通知，或者额外支付一个月工资后解除。\n……\n第十三条 争议解决\n双方发生劳动争议的，应当先向劳动争议仲裁委员会申请仲裁。"}
  ]},
  {document_id:"doc_002_001",file_name:"解除劳动合同通知书.pdf",pages:[
    {page:1,text:"解除劳动合同通知书\n\n测试员工01：\n\n因公司业务调整，经公司研究决定，自2024年1月15日起解除与你的劳动合同。\n\n请你于收到本通知后3个工作日内办理离职手续。\n\n广州科技有限公司\n2024年1月10日"}
  ]},
  {document_id:"doc_003_001",file_name:"工资流水.pdf",pages:[
    {page:1,text:"广州科技有限公司工资单\n姓名：测试员工01  月份：2023年1月至2023年12月\n\n2023-01：15,000元  2023-02：15,000元  2023-03：15,000元\n2023-04：15,000元  2023-05：15,000元  2023-06：15,000元\n2023-07：15,000元  2023-08：15,000元  2023-09：15,000元\n2023-10：15,000元  2023-11：15,000元  2023-12：15,000元\n\n2023年12月工资于2024年1月15日发放。"}
  ]}
];

async function api(path, options={}) {
  const response = await fetch(`/api/api/v1${path}`, {headers:{"Content-Type":"application/json",...(options.headers||{})},...options});
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) throw new Error(data?.detail || data?.error?.message || `HTTP ${response.status}`);
  return data;
}
function notice(message, isError=true){$("notice").textContent=message;$("notice").classList.toggle("hidden",!message);$("notice").style.background=isError?"#f4dfdc":"#e2eee6";}
function reset(){state.runId=null;state.result=null;state.events=[];state.demoDocuments=null;clearInterval(state.timer);$("createView").classList.remove("hidden");$("runView").classList.add("hidden");$("materialHint").textContent="手工粘贴内容会作为第 1 页材料提交。";notice("");setDefaults();}
function setDefaults(){const stamp=new Date().toISOString().replace(/\D/g,"").slice(0,14);$("caseId").value=`case_${stamp}`;}
async function health(){try{await api("/ready");$("serviceDot").classList.add("online");$("serviceText").textContent="服务正常";}catch(e){$("serviceText").textContent="服务不可用";notice(e.message);}}
function highlightAgent(agent,status){document.querySelectorAll(".agent-steps li").forEach((el)=>{el.classList.remove("active","done");const order=["case_understanding_agent","legal_research_agent","case_strategy_agent","document_generation_agent"];const current=order.indexOf(agent);const index=order.indexOf(el.dataset.agent);if(index<current||status==="completed")el.classList.add("done");else if(index===current)el.classList.add("active");});}
function renderEvents(events){$("eventCount").textContent=`${events.length} 个事件`;$("timeline").innerHTML=events.length?events.map((event,i)=>`<div class="timeline-item"><span class="timeline-index">${String(i+1).padStart(2,"0")}</span><div><b>${agentLabels[event.agent]||event.agent||"Orchestrator"}</b><small>${event.node||event.phase||"状态更新"}</small></div><span class="timeline-time">${event.time?new Date(event.time).toLocaleTimeString():""}</span></div>`).join(""):'<p class="empty">等待 Agent 事件…</p>';}
function renderStatus(data){state.status=data.status;$("runStatus").textContent=statusLabels[data.status]||data.status;highlightAgent(data.current_agent,data.status);const review=data.status==="waiting_fact_review"||data.status==="waiting_strategy_review";$("reviewCard").classList.toggle("hidden",!review);if(review){const facts=data.status==="waiting_fact_review";$("reviewTitle").textContent=facts?"确认案件事实":"确认案件策略";$("reviewCopy").textContent=facts?"案件理解已完成，请确认事实基线后进入法律研究。":"法律研究和证据矩阵已完成，请确认策略后生成文书。";$("approveButton").dataset.action=facts?"approve_facts":"approve_strategy";}if(["completed","waiting_fact_review","waiting_strategy_review","waiting_human_intervention","failed","cancelled"].includes(data.status))loadResult();if(data.status==="failed")notice((data.errors||[]).join("；")||"工作流执行失败");}
async function poll(){if(!state.runId)return;try{const response=await api(`/case-runs/${state.runId}`);renderStatus(response.data);await loadResult(false);}catch(e){notice(e.message);}}
async function loadResult(show=true){if(!state.runId)return;try{const response=await api(`/case-runs/${state.runId}/result`);state.result=response.data;state.events=response.data.workflow_events||[];renderEvents(state.events);$("factMetric").textContent=(response.data.facts||[]).length;$("issueMetric").textContent=(response.data.issues||[]).length;$("researchMetric").textContent=(response.data.legal_research||[]).length;$("draftMetric").textContent=(response.data.drafts||[]).length;if(show||response.data.status==="completed"){$("resultPanel").classList.remove("hidden");renderResult();}}catch(e){if(show)notice(e.message);}}
function esc(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function tags(items, extra=""){return (items||[]).map(x=>`<span class="tag ${extra}">${esc(x)}</span>`).join("");}
function planCard(plan,label,klass){if(!plan)return"";return `<article class="result-item ${klass}"><p class="eyebrow">${label}</p><h4>${esc(plan.name||"未命名方案")}</h4><p>${esc(plan.description)}</p><p><b>请求：</b>${esc((plan.claims||[]).join("；")||"未列出")}</p><p><b>证据摘要：</b>${esc((plan.evidence_summary||[]).join("；")||"未列出")}</p><p><b>风险等级：</b>${esc(plan.risk_level)}</p><p><b>对方可能抗辩：</b>${esc((plan.opponent_defenses||[]).join("；")||"未列出")}</p></article>`;}
function renderResult(){
  const r=state.result;if(!r)return;let html="";
  if(state.tab==="facts") html=`<div class="truth-note">事实引用仅展示通过“文档 ID + 页码存在 + 原文逐字包含”校验的来源；无有效来源的事实会降为材料不足。</div><div class="result-list">${(r.facts||[]).map(f=>`<article class="result-item"><h4>${esc(f.statement)}</h4><p>状态：${esc(f.status)} · 置信度：${Math.round((f.confidence||0)*100)}%</p>${(f.sources||[]).map(s=>`<div class="quote">${esc(s.document_id)} · 第 ${s.page} 页<br>“${esc(s.quote)}”</div>`).join("")||'<div class="tag-list"><span class="tag gap">无通过校验的原文来源</span></div>'}</article>`).join("")||"<p>暂无事实</p>"}</div>`;
  if(state.tab==="evidence") html=`<div class="truth-note">每一行对应仓库 EvidenceItem：争点/诉求 → 待证事实 → 证据及页码 → 缺失材料。证据矩阵当前不保存逐字 quote，逐字原文在“事实与原文”页展示。</div><div class="result-list">${(r.evidence_matrix||[]).map(x=>`<article class="result-item"><div class="mapping-path">${esc(x.issue_id)} · ${esc(x.claim)} → ${esc(x.fact_to_prove)}</div><div class="tag-list">${(x.evidence||[]).map(e=>`<span class="tag">${esc(e.evidence_name)}｜${esc(e.document_id)} P${e.page}</span>`).join("")||'<span class="tag gap">暂无证据</span>'}</div><div class="tag-list">${tags(x.missing_materials,"gap")}</div><p>证据状态：${esc(x.status)}${(x.risks||[]).length?` · 风险：${esc(x.risks.join("；"))}`:""}</p></article>`).join("")||"<p>暂无证据矩阵</p>"}</div>`;
  if(state.tab==="research") html=`<div class="result-list">${(r.legal_research||[]).map(x=>`<article class="result-item"><h4>${esc(x.issue_id||"法律研究")}</h4><p>${esc(x.conclusion)}</p><p>引用校验：${x.citations_valid?"通过":"未通过"}</p>${(x.favorable_sources||[]).map(c=>`<div class="quote"><b>${esc(c.title)}</b><br>${esc(c.quote)}</div>`).join("")}</article>`).join("")||"<p>暂无研究结果</p>"}</div>`;
  if(state.tab==="strategy") html=r.strategy?`<div class="truth-note">策略是待律师审核的结构化建议，不是胜诉预测或自动法律结论。</div><div class="strategy-grid">${planCard(r.strategy.primary_plan,"PRIMARY · 主策略","strategy-primary")}${(r.strategy.alternative_plans||[]).map((p,i)=>planCard(p,`ALTERNATIVE ${i+1} · 备选策略`,"strategy-alt")).join("")}</div><div class="tag-list">${tags(r.strategy.pending_confirmations,"gap")}</div>`:"<p>暂无策略</p>";
  if(state.tab==="review"){const d=r.review?.deterministic,s=r.review?.semantic;html=`<div class="truth-note">工作流在事实确认与策略确认两个节点真实中断；审核意见随 resume 请求写入运行警告。下方质量复核是文书生成后的自动复核，不等同于律师批准。</div><div class="result-list"><article class="result-item"><h4>当前人工审核状态</h4><p>${esc(statusLabels[r.status]||r.status)}</p></article><article class="result-item"><h4>确定性复核</h4><p>${d?`结果：${d.passed?"通过":"未通过"}；发现 ${(d.findings||[]).length} 项`:"尚未执行"}</p></article><article class="result-item"><h4>语义复核</h4><p>${s?`结果：${s.passed?"通过":"未通过"}；发现 ${(s.findings||[]).length} 项`:"尚未执行"}</p></article></div>`;}
  if(state.tab==="drafts") html=(r.drafts||[]).map(d=>`<article class="result-item"><h4>${esc(d.document_type)}</h4><div class="document">${esc(d.full_text)}</div></article>`).join("")||"<p>暂无文书</p>";
  if(state.tab==="trace") html=`<div class="result-list">${(r.trace||[]).map(t=>`<article class="result-item"><h4>${esc(t.node)}</h4><p>${esc(t.status)} · ${Math.round(t.duration_ms||0)} ms · ${esc(t.model||"无模型")}</p><small>Prompt ${esc(t.prompt_version)} · Retry ${t.retry_count||0} · Token ${t.token_usage||0}</small></article>`).join("")||"<p>暂无追踪信息</p>"}</div>`;
  $("resultContent").innerHTML=html;
}

$("caseForm").addEventListener("submit",async(e)=>{e.preventDefault();notice("");const caseId=$("caseId").value.trim();const documents=state.demoDocuments||[{document_id:`doc_${caseId.replace(/\W/g,"_")}`,file_name:"案件材料.txt",pages:[{page:1,text:$("materialText").value.trim()}]}];const payload={case_id:caseId,case_type:"labor_dispute",client_role:$("clientRole").value,client_goal:$("clientGoal").value.trim(),documents,idempotency_key:`web_${caseId}`};try{const response=await api("/case-runs",{method:"POST",body:JSON.stringify(payload)});if(response.status==="error")throw new Error(response.error?.message||"创建失败");state.runId=response.data.run_id;$("createView").classList.add("hidden");$("runView").classList.remove("hidden");$("runTitle").textContent=caseId;$("runId").textContent=state.runId;state.timer=setInterval(poll,1500);poll();}catch(err){notice(err.message);}});
$("approveButton").addEventListener("click",async()=>{const button=$("approveButton");button.disabled=true;button.textContent="处理中…";try{const response=await api(`/case-runs/${state.runId}/resume`,{method:"POST",body:JSON.stringify({action:button.dataset.action,comments:$("reviewComment").value})});renderStatus({status:response.data.status,current_agent:button.dataset.action==="approve_facts"?"case_strategy_agent":"document_generation_agent"});$("reviewComment").value="";}catch(e){notice(e.message);}finally{button.disabled=false;button.textContent="确认并继续";poll();}});
$("cancelButton").addEventListener("click",async()=>{if(!confirm("确认取消当前任务？"))return;try{await api(`/case-runs/${state.runId}/resume`,{method:"POST",body:JSON.stringify({action:"cancel"})});poll();}catch(e){notice(e.message);}});
document.querySelectorAll(".tab").forEach(tab=>tab.addEventListener("click",()=>{document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));tab.classList.add("active");state.tab=tab.dataset.tab;renderResult();}));
$("refreshButton").addEventListener("click",()=>loadResult(true));$("newCaseButton").addEventListener("click",reset);
$("loadDemoButton").addEventListener("click",()=>{
  state.demoDocuments=demoDocuments;
  $("caseId").value=`demo_eval_001_${new Date().toISOString().replace(/\D/g,"").slice(8,14)}`;
  $("clientRole").value="employee";
  $("clientGoal").value="请求支付违法解除赔偿金和拖欠工资";
  $("materialText").value=demoDocuments.map(d=>`${d.file_name}（${d.pages.length} 页）\n${d.pages.map(p=>`P${p.page}: ${p.text}`).join("\n")}`).join("\n\n");
  $("materialHint").textContent="已载入 lexflow_agent/data/eval/dev_set/sample_cases.json 的 eval_001 前三份分页材料。";
  notice("已载入仓库合成脱敏样例 eval_001；点击“开始分析”将按原文档 ID 和页码提交。",false);
});
setDefaults();health();
