"""脱敏函数单元测试"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestSecurity:
    """测试 base.security 脱敏函数"""

    def test_mask_phone(self):
        from base.security import mask_phone
        assert mask_phone("13812345678") == "138****5678"
        assert mask_phone("手机 13900139000 号") == "手机 139****9000 号"
        # 短数字不应脱敏
        assert mask_phone("12345") == "12345"

    def test_mask_id_card(self):
        from base.security import mask_id_card
        assert mask_id_card("110101199001011234") == "110101********1234"
        assert mask_id_card("11010119900101123X") == "110101********123X"

    def test_mask_api_key(self):
        from base.security import mask_api_key
        key = "sk-7a89cb588d844aeeb12a3b990dd20190"
        result = mask_api_key(key)
        assert result == "sk-****0190"
        assert "7a89" not in result

    def test_mask_email(self):
        from base.security import mask_email
        assert mask_email("user@example.com") == "u***@example.com"
        assert mask_email("zhangsan@test.cn") == "z***@test.cn"

    def test_mask_sensitive_text(self):
        from base.security import mask_sensitive_text
        text = "user: Zhang San, phone 13812345678, id 110101199001011234"
        result = mask_sensitive_text(text)
        assert "138****5678" in result
        assert "110101********1234" in result
        assert "Zhang San" in result  # 姓名不应被脱敏
