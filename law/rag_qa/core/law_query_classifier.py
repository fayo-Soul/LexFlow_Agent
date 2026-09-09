# 1. 导入依赖包
import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from base import setup_logger, Config

# 2. 处理项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, rag_qa_path)
sys.path.insert(0, project_root)


# 3. 定义法律领域查询分类器
class LawQueryClassifier:
    """
    基于 bert_law_classifier 的法律领域多标签分类器。
    输出顺序与 config.ini 中 valid_sources 保持一致，默认 fallback 为 5 大法律领域。
    """

    def __init__(self, model_path=None, threshold=0.7):
        self.logger = setup_logger("LawQueryClassifier")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")

        # 模型路径默认指向训练脚本保存的位置（通过 Config 可配置）
        if model_path is None:
            model_path = Config().BERT_CLASSIFIER_DIR
        self.model_path = model_path
        self.threshold = threshold

        # 从 config.ini 读取类别顺序，失败则使用硬编码 fallback
        self.LABEL_NAMES = self._load_label_names()

        self.tokenizer = None
        self.model = None
        self.load_model()

    @staticmethod
    def _load_label_names():
        try:
            sources = Config().VALID_SOURCES
            if isinstance(sources, str):
                sources = eval(sources)
            if isinstance(sources, (list, tuple)) and len(sources) == 5:
                return list(sources)
        except Exception:
            pass
        return ["刑法", "民法", "劳动法", "行政法", "其他法"]

    def load_model(self):
        if not os.path.exists(self.model_path):
            self.logger.error(f"法律分类器模型不存在：{self.model_path}")
            return

        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # 校验模型标签映射与期望顺序是否一致
        config_id2label = getattr(self.model.config, "id2label", {})
        expected_id2label = {i: name for i, name in enumerate(self.LABEL_NAMES)}
        if config_id2label != expected_id2label:
            self.logger.warning(
                f"模型 config 中的 id2label {config_id2label} 与期望 {expected_id2label} 不一致，"
                f"将以代码中 LABEL_NAMES 的顺序为准：{self.LABEL_NAMES}"
            )
        else:
            self.logger.info(f"模型标签映射校验通过：{self.LABEL_NAMES}")

        self.logger.info(f"加载法律分类器模型：{self.model_path}")

    # 简单关键词规则兜底：BERT 模型误判高频场景时，用规则强纠正
    KEYWORD_RULES = {
        "行政法": ["交警", "高速", "无证驾驶", "行政处罚", "行政拘留", "罚款", "违章",
                 "违法停车", "驾驶证", "行驶证", "城管", "工商", "税务", "海关", "超载"],
        "民法": ["离婚", "抚养权", "抚养费", "继承", "遗嘱", "合同", "借款", "债务",
               "房产", "物业", "婚姻", "彩礼", "侵权", "赔偿", "违约金", "借条"],
        "劳动法": ["工资", "劳动合同", "劳动仲裁", "工伤", "社保", "五险一金", "辞退",
                 "离职", "加班", "加班费", "经济补偿", "赔偿金"],
        "刑法": ["犯罪", "盗窃", "诈骗", "抢劫", "杀人", "故意伤害", "毒品", "醉驾",
               "刑事拘留", "判刑", "有期徒刑", "无期徒刑", "死刑"],
    }

    # 非法律问题关键词（常见技术/生活/通用问题），命中后直接返回 None
    # 注意：只放较强指示词，避免与普通法律查询词（如“文件”“读取”）冲突
    NON_LEGAL_KEYWORDS = [
        "Python", "Java", "C++", "JavaScript", "JS", "Go", "Rust", "PHP", "Ruby",
        "代码", "程序", "编程", "程序员", "CSV", "Excel", "数据库", "SQL",
        "算法", "函数", "变量", "数组", "列表", "字典", "字符串", "bug",
        "IDE", "Git", "Linux", "Ubuntu", "Windows", "Mac", "浏览器", "网页",
        "服务器", "API", "HTTP", "HTTPS", "JSON", "XML", "YAML", "Docker",
        "Kubernetes", "K8s", "云计算", "大数据", "人工智能", "机器学习",
        "怎么做饭", "怎么做菜", "天气", "今天星期几", "你好", "谢谢", "再见",
    ]

    def _rule_predict(self, query):
        """基于关键词规则预测领域，命中唯一规则时返回该领域，否则返回 None"""
        # 非法律问题快速兜底（较强指示词）
        if any(kw in query for kw in self.NON_LEGAL_KEYWORDS):
            self.logger.info(f"关键词规则命中非法律问题：{query}")
            return None

        matched = []
        for label, keywords in self.KEYWORD_RULES.items():
            if any(kw in query for kw in keywords):
                matched.append(label)
        if len(matched) == 1:
            self.logger.info(f"关键词规则命中：{query} -> {matched[0]}")
            return matched[0]
        if len(matched) > 1:
            self.logger.info(f"关键词规则命中多个领域 {matched}，回退到模型预测")
        return None

    def predict(self, query):
        """
        预测查询所属的法律领域。
        返回：
            - str: 概率最高的法律领域标签（当最大概率 >= threshold 时）
            - None: 当所有标签概率均低于 threshold 时，视为非法律问题
        """
        label, confidence = self.predict_with_confidence(query)
        return label if confidence >= self.threshold else None

    def predict_with_confidence(self, query):
        """返回最高概率领域及其置信度，供统一查询规划器决定是否调用 LLM。"""
        rule_label = self._rule_predict(query)
        if rule_label:
            return rule_label, 1.0
        if self.model is None or self.tokenizer is None:
            self.logger.error("法律分类器模型未加载")
            return None, 0.0

        encoding = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])
        predicted_label = self.LABEL_NAMES[max_idx]
        self.logger.info(
            f"查询：{query} | 预测概率："
            f"{dict(zip(self.LABEL_NAMES, [round(float(p), 3) for p in probs]))}"
        )
        self.logger.info(f"预测法律领域：{predicted_label}，概率：{max_prob:.4f}")
        return predicted_label, max_prob

    def predict_multi(self, query):
        """
        返回所有概率超过 threshold 的法律领域标签列表（支持多标签）。
        """
        if self.model is None or self.tokenizer is None:
            self.logger.error("法律分类器模型未加载")
            return []

        encoding = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        labels = [self.LABEL_NAMES[i] for i, p in enumerate(probs) if float(p) >= self.threshold]
        self.logger.info(
            f"查询：{query} | 多标签预测：{labels} | "
            f"概率：{dict(zip(self.LABEL_NAMES, [round(float(p), 3) for p in probs]))}"
        )
        return labels


# 4. 测试入口
if __name__ == '__main__':
    classifier = LawQueryClassifier()
    test_queries = [
        "无证驾驶被高速交警大队抓了怎么办",
        "离婚之后孩子的抚养权怎么判",
        "公司拖欠工资三个月了，我能申请劳动仲裁吗",
        "Python 怎么读取 CSV 文件"
    ]
    for q in test_queries:
        result = classifier.predict(q)
        print(f"问题：{q}")
        print(f"预测领域：{result}")
        print("-" * 60)
