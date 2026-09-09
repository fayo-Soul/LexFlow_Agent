"""数据脱敏工具

提供手机号、身份证号、API Key 等敏感信息的脱敏函数。

用法：
    from base.security import mask_sensitive_text, mask_api_key

    clean = mask_sensitive_text("用户手机 13812345678，身份证 110101199001011234")
    # => "用户手机 138****5678，身份证 110101********1234"
"""

import re


def mask_phone(text: str) -> str:
    """脱敏手机号：13812345678 → 138****5678"""
    return re.sub(r"1[3-9]\d{9}", lambda m: m.group(0)[:3] + "****" + m.group(0)[-4:], text)


def mask_id_card(text: str) -> str:
    """脱敏身份证号：110101199001011234 → 110101********1234"""
    return re.sub(r"\d{17}[\dXx]", lambda m: m.group(0)[:6] + "********" + m.group(0)[-4:], text)


def mask_api_key(text: str) -> str:
    """脱敏 API Key：sk-7a89cb588d844aeeb12a3b990dd20190 → sk-****0190"""
    return re.sub(r"(sk-)[a-f0-9]{32}", lambda m: m.group(1) + "****" + m.group(0)[-4:], text)


def mask_email(text: str) -> str:
    """脱敏邮箱：user@example.com → u***@example.com"""
    return re.sub(r"([a-zA-Z0-9])[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                  lambda m: m.group(1) + "***@" + m.group(2), text)


def mask_sensitive_text(text: str) -> str:
    """综合脱敏：身份证 → 手机号 → API Key → 邮箱

    身份证必须先于手机号处理，因为身份证号中可能包含符合手机号模式的子串。
    """
    text = mask_id_card(text)
    text = mask_phone(text)
    text = mask_api_key(text)
    text = mask_email(text)
    return text
