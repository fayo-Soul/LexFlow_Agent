# 语法检查脚本
# 在每个新环境首次运行

set -euo pipefail

echo "=== 语法检查开始 ==="

compile() {
    python -m py_compile "$1" && echo "  OK: $1" || echo "  FAIL: $1"
}

compile lexflow_agent/config/settings.py
compile lexflow_agent/config/config_loader.py
compile lexflow_agent/engine/models/enums.py
compile lexflow_agent/engine/models/case.py
compile lexflow_agent/engine/models/facts.py
compile lexflow_agent/engine/models/evidence.py
compile lexflow_agent/engine/models/issues.py
compile lexflow_agent/engine/models/research.py
compile lexflow_agent/engine/models/strategy.py
compile lexflow_agent/engine/models/document.py
compile lexflow_agent/engine/models/review.py
compile lexflow_agent/engine/models/response.py
compile lexflow_agent/engine/state.py
compile lexflow_agent/engine/resilience/retry.py
compile lexflow_agent/engine/resilience/circuit_breaker.py
compile lexflow_agent/engine/resilience/fallback.py
compile lexflow_agent/engine/cache/semantic_cache.py
compile lexflow_agent/engine/precheck/rule_engine.py
compile lexflow_agent/engine/precheck/intent_classifier.py
compile lexflow_agent/engine/tools/law_client.py
compile lexflow_agent/engine/tools/doc_reader.py
compile lexflow_agent/engine/tools/calculator.py
compile lexflow_agent/engine/nodes/validation.py
compile lexflow_agent/engine/nodes/material.py
compile lexflow_agent/engine/nodes/facts.py
compile lexflow_agent/engine/nodes/issues.py
compile lexflow_agent/engine/nodes/evidence.py
compile lexflow_agent/engine/nodes/legal_research.py
compile lexflow_agent/engine/nodes/strategy.py
compile lexflow_agent/engine/nodes/drafting.py
compile lexflow_agent/engine/nodes/deterministic_review.py
compile lexflow_agent/engine/nodes/semantic_review.py
compile lexflow_agent/engine/graph.py
compile lexflow_agent/engine/gateway/factory.py
compile lexflow_agent/engine/runtime/__init__.py
compile lexflow_agent/api/app.py
compile lexflow_agent/api/routes/case_runs.py
compile lexflow_agent/api/routes/health.py

compile lexflow_agent/tests/unit/test_schemas.py
compile lexflow_agent/tests/unit/test_retry.py
compile lexflow_agent/tests/unit/test_circuit_breaker.py
compile lexflow_agent/tests/unit/test_rule_engine.py
compile lexflow_agent/tests/unit/test_calculator.py

echo ""
echo "=== 语法检查完成 ==="
echo "如果所有显示 OK，说明代码无语法错误"
