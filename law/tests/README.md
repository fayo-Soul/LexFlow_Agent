# tests

项目自动化测试。

## 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 仅运行某测试文件
python -m pytest tests/test_config.py -v
```

## 测试分类

| 文件 | 类型 | 是否依赖外部服务 |
|---|---|---|
| `test_config.py` | 单元测试 | 否 |
| `test_api_health.py` | 集成测试 | 否（仅检查模块加载和启动） |
