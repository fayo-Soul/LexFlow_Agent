"""劳动争议计算工具 - 确定性计算

本模块提供劳动争议相关的确定性计算功能，所有计算都是确定性的（不依赖 LLM）。
计算结果包含：
- 输入参数（用于审计和回溯）
- 计算公式（用于解释计算逻辑）
- 计算结果（精确数值）
- 风险提示（法律合规提示）

适用场景：
- 工作年限计算
- 经济补偿金（N）计算
- 违法解除赔偿金（2N）计算
- 仲裁时效日期提示
- 工资金额差额计算
"""

from __future__ import annotations  # 启用未来类型注解

from datetime import date  # 日期处理
from typing import Optional  # 可选类型提示


class LaborCalculator:
    """劳动争议计算器 - 提供确定性计算功能
    
    所有计算方法都是静态方法，返回包含输入、公式、结果和风险提示的字典。
    这些计算不依赖 LLM，确保计算结果的准确性和可解释性。
    """

    @staticmethod
    def calc_work_years(start: date, end: Optional[date] = None) -> dict:
        """计算工作年限 - 从入职日期到离职日期（或今天）
        
        Args:
            start: 入职日期
            end: 离职日期（默认今天）
        
        Returns:
            包含工作年限的字典（年数、月数、计算公式、风险提示）
        """
        end = end or date.today()  # 如果未提供离职日期，使用今天
        years = (end - start).days / 365.25  # 计算年数（考虑闰年）
        months = int(years * 12)  # 转换为月数
        return {
            "input": {"start": start.isoformat(), "end": end.isoformat()},  # 输入日期
            "formula": "(end - start).days / 365.25 * 12",  # 计算公式
            "result_months": months,  # 结果（月）
            "result_years": round(years, 2),  # 结果（年，保留两位小数）
            "risk": "工作年限跨整年时，计算方式可能因地区规定略有差异",  # 风险提示
        }

    @staticmethod
    def calc_severance_pay(months_of_service: int, avg_monthly_salary: float) -> dict:
        """计算经济补偿金 (N) - 根据工作年限和月平均工资
        
        Args:
            months_of_service: 工作月数
            avg_monthly_salary: 月平均工资
        
        Returns:
            包含经济补偿金的字典（金额、上限金额、风险提示）
        """
        amount = months_of_service * avg_monthly_salary  # 基本计算
        max_months = 12  # 法定上限：12 个月
        capped = min(months_of_service, max_months) * avg_monthly_salary  #  capped 金额
        return {
            "input": {"months": months_of_service, "avg_salary": avg_monthly_salary},  # 输入
            "formula": "months_of_service × avg_monthly_salary",  # 公式
            "result": round(amount, 2),  # 计算结果
            "capped_result": round(capped, 2) if months_of_service > max_months else None,  # 上限结果
            "risk": f"月工资超过当地社平工资3倍时，计算基数上限为社平工资3倍，年限上限{max_months}个月",  # 风险提示
        }

    @staticmethod
    def calc_wrongful_termination_penalty(severance_pay: float) -> dict:
        """计算违法解除赔偿金 (2N) - 经济补偿金的 2 倍
        
        Args:
            severance_pay: 经济补偿金金额
        
        Returns:
            包含违法解除赔偿金的字典（金额、计算公式、风险提示）
        """
        amount = severance_pay * 2  # 2 倍计算
        return {
            "input": {"severance_pay": severance_pay},  # 输入
            "formula": "severance_pay × 2",  # 公式
            "result": round(amount, 2),  # 结果
            "risk": "违法解除赔偿金为经济补偿金的2倍。是否构成违法解除需结合案件事实判断。",  # 风险提示
        }

    @staticmethod
    def calc_arbitration_deadline(termination_date: date) -> dict:
        """计算仲裁时效日期提示 - 劳动争议仲裁时效为 1 年
        
        Args:
            termination_date: 劳动关系终止日期
        
        Returns:
            包含仲裁时效的字典（截止日期、剩余天数、风险提示）
        """
        from datetime import timedelta  # 时间差处理
        deadline = termination_date + timedelta(days=365)  # 1 年时效
        return {
            "input": {"termination_date": termination_date.isoformat()},  # 输入日期
            "formula": "termination_date + 1年",  # 公式
            "deadline": deadline.isoformat(),  # 截止日期
            "days_remaining": (deadline - date.today()).days,  # 剩余天数
            "risk": "劳动争议仲裁时效为1年，自知道或应当知道权利被侵害之日起计算。拖欠劳动报酬的，劳动关系终止时起算。",  # 风险提示
        }

    @staticmethod
    def calc_salary_diff(expected: float, actual: float) -> dict:
        """计算工资金额差额 - 预期工资与实际工资之差
        
        Args:
            expected: 预期工资金额
            actual: 实际工资金额
        
        Returns:
            包含工资差额的字典（差额、计算公式、风险提示）
        """
        diff = expected - actual  # 差额计算
        return {
            "input": {"expected": expected, "actual": actual},  # 输入
            "formula": "expected - actual",  # 公式
            "result": round(diff, 2),  # 结果
            "risk": "需确认工资构成是否包含津贴、补贴、奖金等。",  # 风险提示
        }


# 全局单例 - 提供便捷的计算器访问
calculator = LaborCalculator()