"""配置读取单元测试"""

import os
import sys
import pytest

# 确保能找到项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestConfig:
    """测试 base.config.Config 的配置读取"""

    def test_config_defaults(self):
        """不依赖外部文件，测试默认值"""
        from base.config import Config

        # 临时清除 .env 中已加载的值
        backup = {}
        for key in ["MYSQL_HOST", "MYSQL_PORT", "REDIS_PORT", "MILVUS_HOST",
                     "PARENT_CHUNK_SIZE", "CHILD_CHUNK_SIZE"]:
            backup[key] = os.environ.pop(key, None)

        try:
            cfg = Config(config_file="nonexistent.ini")
            assert cfg.MYSQL_HOST == "127.0.0.1", f"got {cfg.MYSQL_HOST}"
            assert cfg.MYSQL_PORT == "3306"
            assert cfg.REDIS_PORT == "6379"
            assert cfg.MILVUS_HOST == "127.0.0.1"
            assert cfg.PARENT_CHUNK_SIZE == 512
            assert isinstance(cfg.PARENT_CHUNK_SIZE, int)
            assert cfg.CHILD_CHUNK_SIZE == 128
        finally:
            # 恢复环境变量
            for key, val in backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_config_from_ini(self):
        """测试从 config.ini 读取"""
        ini_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.ini")
        if not os.path.exists(ini_path):
            pytest.skip("config.ini not found")

        from base.config import Config

        cfg = Config(config_file=ini_path)
        assert cfg.MYSQL_HOST is not None
        assert len(cfg.MYSQL_HOST) > 0

    def test_env_var_override(self):
        """测试环境变量优先级高于 ini"""
        os.environ["_TEST_MYSQL_HOST"] = "env_override_test"
        from base.config import get_config
        import configparser

        config = configparser.ConfigParser()
        config.read_string("[mysql]\nhost = ini_value\n")

        result = get_config(config, "mysql", "host", "_TEST_MYSQL_HOST", "default")
        assert result == "env_override_test"
        del os.environ["_TEST_MYSQL_HOST"]

    def test_get_config_fallback(self):
        """测试获取值时的三层回退"""
        from base.config import get_config
        import configparser

        config = configparser.ConfigParser()
        config.read_string("[test]\n")

        # env 无，ini 无 → 返回默认值
        result = get_config(config, "test", "nonexistent", "_NONEXISTENT_ENV", "fallback_val")
        assert result == "fallback_val"
