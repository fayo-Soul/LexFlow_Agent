"""单元测试: 劳动争议计算器"""

from datetime import date
from lexflow_agent.engine.tools.calculator import calculator


class TestCalculator:
    def test_work_years(self):
        result = calculator.calc_work_years(date(2020, 1, 1), date(2025, 1, 1))
        assert result["result_years"] >= 4.9
        assert result["result_years"] <= 5.1

    def test_severance_pay(self):
        result = calculator.calc_severance_pay(12, 5000)
        assert result["result"] == 60000.0

    def test_wrongful_termination_penalty(self):
        result = calculator.calc_wrongful_termination_penalty(60000)
        assert result["result"] == 120000.0

    def test_arbitration_deadline(self):
        result = calculator.calc_arbitration_deadline(date(2025, 1, 1))
        assert result["deadline"] == "2026-01-01"

    def test_salary_diff(self):
        result = calculator.calc_salary_diff(10000, 8000)
        assert result["result"] == 2000.0

    def test_negative_diff(self):
        result = calculator.calc_salary_diff(8000, 10000)
        assert result["result"] == -2000.0
