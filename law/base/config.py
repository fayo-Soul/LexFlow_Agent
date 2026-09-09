# 1. 导包
import os
import configparser
from dotenv import load_dotenv

# 2. 加载 .env 文件（优先于 config.ini）
load_dotenv()

# 3. 获取 config_file_path 配置文件绝对路径
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir_path)
config_file_path = os.path.join(project_root, "config.ini")


# 4. 辅助函数：环境变量优先 → config.ini → 默认值
def get_config(config, section, key, env_name, fallback=None):
    """获取配置，优先级：环境变量 > config.ini > 默认值"""
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value
    return config.get(section, key, fallback=fallback)


# 5. 封装 Config 类，用于读取文件信息 => 分板块进行读取
class Config(object):
    def __init__(self, config_file=config_file_path):
        # 5.1 获取 config 解析器对象
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding="utf-8")

        # 5.2 获取 mysql 板块信息
        self.MYSQL_HOST = get_config(self.config, "mysql", "host", "MYSQL_HOST", "127.0.0.1")
        self.MYSQL_PORT = get_config(self.config, "mysql", "port", "MYSQL_PORT", "3306")
        self.MYSQL_USER = get_config(self.config, "mysql", "user", "MYSQL_USER", "root")
        self.MYSQL_PASSWORD = get_config(self.config, "mysql", "password", "MYSQL_PASSWORD", "")
        self.MYSQL_DATABASE = get_config(self.config, "mysql", "database", "MYSQL_DATABASE", "fzt_law")
        self.MYSQL_LAW_QA_TABLE = get_config(self.config, "mysql", "law_qa_table", "MYSQL_LAW_QA_TABLE", "fzt_qa")

        # 5.3 获取 redis 板块信息
        self.REDIS_HOST = get_config(self.config, "redis", "host", "REDIS_HOST", "127.0.0.1")
        self.REDIS_PORT = get_config(self.config, "redis", "port", "REDIS_PORT", "6379")
        self.REDIS_PASSWORD = get_config(self.config, "redis", "password", "REDIS_PASSWORD", "")
        self.REDIS_DB = get_config(self.config, "redis", "db", "REDIS_DB", "1")

        # 5.4 获取 milvus 板块信息
        self.MILVUS_HOST = get_config(self.config, "milvus", "host", "MILVUS_HOST", "127.0.0.1")
        self.MILVUS_PORT = get_config(self.config, "milvus", "port", "MILVUS_PORT", "19530")
        self.MILVUS_DATABASE_NAME = get_config(self.config, "milvus", "database_name", "MILVUS_DATABASE_NAME", "fzt_law")
        self.MILVUS_COLLECTION_NAME = get_config(self.config, "milvus", "collection_name", "MILVUS_COLLECTION_NAME", "fzt_legal_articles")

        # 5.4a 案例库 Milvus 配置
        self.CASE_MILVUS_DATABASE_NAME = get_config(self.config, "milvus", "case_database_name", "CASE_MILVUS_DATABASE_NAME", "fzt_case")
        self.CASE_MILVUS_COLLECTION_NAME = get_config(self.config, "milvus", "case_collection_name", "CASE_MILVUS_COLLECTION_NAME", "fzt_case_hybrid")

        # 5.5 获取 llm 板块信息
        self.LLM_MODEL = get_config(self.config, "llm", "model", "LLM_MODEL", "qwen3.7-plus")
        self.DASHSCOPE_API_KEY = get_config(self.config, "llm", "dashscope_api_key", "DASHSCOPE_API_KEY", "")
        self.DASHSCOPE_BASE_URL = get_config(self.config, "llm", "dashscope_base_url", "DASHSCOPE_BASE_URL",
                                            "https://dashscope.aliyuncs.com/compatible-mode/v1")

        # 5.6 获取 retrieval 板块信息
        self.PARENT_CHUNK_SIZE = int(get_config(self.config, "retrieval", "parent_chunk_size",
                                                "PARENT_CHUNK_SIZE", "512"))
        self.CHILD_CHUNK_SIZE = int(get_config(self.config, "retrieval", "child_chunk_size",
                                               "CHILD_CHUNK_SIZE", "128"))
        self.CHUNK_OVERLAP = int(get_config(self.config, "retrieval", "chunk_overlap", "CHUNK_OVERLAP", "50"))
        self.RETRIEVAL_K = int(get_config(self.config, "retrieval", "retrieval_k", "RETRIEVAL_K", "5"))
        self.CANDIDATE_M = int(get_config(self.config, "retrieval", "candidate_m", "CANDIDATE_M", "2"))

        # 5.7 获取 app 板块信息
        self.VALID_SOURCES = get_config(self.config, "app", "valid_sources", "VALID_SOURCES",
                                        '["刑法", "民法", "劳动法", "行政法", "其他法"]')
        self.LEGAL_QA_FILE = get_config(self.config, "app", "legal_qa_file", "LEGAL_QA_FILE", "法律问答.txt")
        self.CUSTOMER_SERVICE_PHONE = get_config(self.config, "app", "customer_service_phone",
                                                 "CUSTOMER_SERVICE_PHONE", "13575008699")

        # 5.8 API 安全配置
        self.API_TOKEN = get_config(self.config, "api", "token", "API_TOKEN", "")
        self.RESEARCHER_API_TOKEN = get_config(
            self.config, "api", "researcher_token", "RESEARCHER_API_TOKEN", ""
        )
        cors_origins = get_config(
            self.config,
            "api",
            "cors_origins",
            "CORS_ORIGINS",
            "http://127.0.0.1:8000,http://localhost:8000",
        )
        self.CORS_ORIGINS = [
            item.strip() for item in cors_origins.split(",") if item.strip()
        ]
        self.RATE_LIMIT_PER_MINUTE = int(
            get_config(
                self.config,
                "api",
                "rate_limit_per_minute",
                "RATE_LIMIT_PER_MINUTE",
                "60",
            )
        )
        self.MAX_UPLOAD_MB = int(
            get_config(self.config, "api", "max_upload_mb", "MAX_UPLOAD_MB", "20")
        )
        self.DOCUMENT_REGISTRY_BACKEND = get_config(
            self.config,
            "api",
            "document_registry_backend",
            "DOCUMENT_REGISTRY_BACKEND",
            "mysql",
        )
        self.INDEX_QUEUE_ENABLED = get_config(
            self.config, "api", "index_queue_enabled", "INDEX_QUEUE_ENABLED", "true"
        ).lower() == "true"
        self.SESSION_SIGNING_KEY = get_config(
            self.config,
            "api",
            "session_signing_key",
            "SESSION_SIGNING_KEY",
            self.API_TOKEN,
        )
        self.ALLOW_INTERNAL_TO_EXTERNAL_LLM = get_config(
            self.config,
            "api",
            "allow_internal_to_external_llm",
            "ALLOW_INTERNAL_TO_EXTERNAL_LLM",
            "false",
        ).lower() == "true"

        # 5.9 模型路径配置
        self.MODEL_DIR = get_config(self.config, "model", "model_dir", "MODEL_DIR", "rag_qa/models")
        self.BERT_CLASSIFIER_DIR = get_config(self.config, "model", "bert_classifier_dir", "BERT_CLASSIFIER_DIR",
                                              os.path.join(project_root, "rag_qa", "models", "bert_law_classifier"))
        self.BGE_M3_DIR = get_config(self.config, "model", "bge_m3_dir", "BGE_M3_DIR",
                                     os.path.join(project_root, "rag_qa", "models", "bge-m3"))
        self.BGE_RERANKER_DIR = get_config(self.config, "model", "bge_reranker_dir", "BGE_RERANKER_DIR",
                                           os.path.join(project_root, "rag_qa", "models", "bge-reranker-large"))

        # 5.10 日志配置
        self.LOG_FILE = get_config(self.config, "logger", "log_file", "LOG_FILE", "logs/app.log")


# 编写 __name__ == "__main__" 测试代码
if __name__ == '__main__':
    config = Config()
    print(f"MYSQL_HOST: {config.MYSQL_HOST}")
    print(f"DASHSCOPE_API_KEY: {'***' if config.DASHSCOPE_API_KEY else '空'}")
    print(f"VALID_SOURCES: {config.VALID_SOURCES}")
    print(f"CUSTOMER_SERVICE_PHONE: {config.CUSTOMER_SERVICE_PHONE}")
    print(f"BERT_CLASSIFIER_DIR: {config.BERT_CLASSIFIER_DIR}")
    print(f"BGE_M3_DIR: {config.BGE_M3_DIR}")
    print(f"BGE_RERANKER_DIR: {config.BGE_RERANKER_DIR}")
