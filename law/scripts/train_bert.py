import json
import os
import sys
import logging
import torch
import numpy as np

# 允许从 scripts/ 目录运行，也能从项目根目录运行
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from base import Config


def setup_logger(name, log_file="train_law.log"):
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(fmt="%(asctime)s-%(levelname)s-%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    if not logger.hasHandlers():
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


logger = setup_logger("LawTrain")


def load_label_names():
    """从 config.ini 读取法律类别顺序，失败则使用硬编码 fallback"""
    try:
        sources = Config().VALID_SOURCES
        if isinstance(sources, str):
            sources = eval(sources)
        if isinstance(sources, (list, tuple)) and len(sources) == 5:
            return list(sources)
    except Exception as e:
        logger.warning(f"从 config.ini 读取类别失败：{e}，使用默认类别")
    return ["刑法", "民法", "劳动法", "行政法", "其他法"]


LABEL_NAMES = load_label_names()
NUM_LABELS = len(LABEL_NAMES)

PRETRAINED_MODEL_DIR = "rag_qa/models/bert-base-chinese"
DATA_FILE = "./preprocessed_data.jsonl"
OUTPUT_MODEL_DIR = "rag_qa/models/bert_law_classifier"


# 【已修复 Bug】确保在 32 vCPU 多进程数据加载（num_workers=6）时，Tensor 能够被正确克隆和切片
class LawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)


def load_data(data_file):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据集文件不存在：{data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    texts = [item['question'] for item in data]
    label_vecs = [item['label_vec'] for item in data]
    logger.info(f"数据加载完成，共 {len(data)} 条")
    return texts, label_vecs


def preprocess(tokenizer, texts, label_vecs):
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,  # 根据法律短文本咨询设为128。若长文本多，可改为256或512
        return_tensors="pt"
    )
    labels = torch.tensor(label_vecs, dtype=torch.float)
    return encodings, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # 多标签分类采用 Sigmoid 激活
    predictions = (probs >= 0.5).astype(int)

    accuracy = np.mean(np.all(predictions == labels, axis=1))
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    return {"subset_accuracy": accuracy, "f1_micro": f1_micro, "f1_macro": f1_macro}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    logger.info(f"使用设备: {device}, GPU数量: {n_gpus}")
    if n_gpus > 0:
        logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU总显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_DIR,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    logger.info(f"模型加载完成: {PRETRAINED_MODEL_DIR}")

    texts, label_vecs = load_data(DATA_FILE)
    train_texts, val_texts, train_vecs, val_vecs = train_test_split(
        texts, label_vecs, test_size=0.2, random_state=42
    )
    logger.info(f"训练集: {len(train_texts)} 条, 验证集: {len(val_texts)} 条")

    train_enc, train_lab = preprocess(tokenizer, train_texts, train_vecs)
    val_enc, val_lab = preprocess(tokenizer, val_texts, val_vecs)
    train_dataset = LawDataset(train_enc, train_lab)
    val_dataset = LawDataset(val_enc, val_lab)

    # 实际总 Batch Size = 单卡 Batch Size (64) * 梯度累积 (1) * GPU数量 (2) = 128
    effective_batch = 64 * 1 * max(1, n_gpus)
    logger.info(f"有效batch_size: {effective_batch}")

    # ==================== 专为双卡 A10 (24GB) 优化的超参数配置 ====================
    training_args = TrainingArguments(
        output_dir="./bert_law_results",
        num_train_epochs=5,  # 5个轮次足够收敛且能配合最优权重回滚
        per_device_train_batch_size=96,  # 单卡 24GB 显存完美吃下 64 Batch Size
        per_device_eval_batch_size=128,  # 评估时不计算梯度，直接开到 128 提高效率
        gradient_accumulation_steps=2,  # 显存充足，设为1直出梯度，不使用梯度累积牺牲速度
        learning_rate=1.5e-5,  # 对应总 Batch Size=128 的经典稳定学习率
        weight_decay=0.01,  # L2 正则化避免模型死记硬背关键词
        warmup_ratio=0.1,  # 前 10% 的步数线性升温，保护预训练权重
        logging_dir="./bert_law_logs",
        logging_steps=20,  # 每 20 步打印一次日志
        evaluation_strategy="epoch",
        save_strategy="epoch",

        # 评测标准优化：以更符合多标签分类业务表现的 微平均F1值 作为历史最佳保存依据
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        save_total_limit=1,  # 仅保留最佳的一个 Checkpoint，节约硬盘

        fp16=True,  # 开启 A10 的 FP16 混合精度硬件加速（速度翻倍）
        gradient_checkpointing=False,  # 关闭梯度检查点。24GB显存充足，不需要临时重算梯度，速度最大化
        dataloader_num_workers=4,  # 基于机器 32 vCPU 的规格，每张卡配6个多进程读取（共12进程），喂饱 GPU
        dataloader_pin_memory=True,
        report_to="tensorboard",
    )
    # ========================================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    logger.info("开始训练模型...")
    trainer.train()

    # 自动加载并保存整个训练周期中 `f1_micro` 表现最好的模型
    logger.info("保存最优模型...")
    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    logger.info(f"最优模型保存至: {OUTPUT_MODEL_DIR}")

    # 用最终的最佳权重在验证集上跑一次详细的分类报告
    logger.info("最终评估模型...")
    predictions = trainer.predict(val_dataset)
    probs = 1 / (1 + np.exp(-predictions.predictions))
    pred_labels = (probs >= 0.5).astype(int)
    true_labels = np.array(val_vecs)

    report = classification_report(true_labels, pred_labels, target_names=LABEL_NAMES, zero_division=0)
    logger.info(f"分类报告:\n{report}")

    with open("eval_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("评估报告已保存至 eval_report.txt")


if __name__ == '__main__':
    main()