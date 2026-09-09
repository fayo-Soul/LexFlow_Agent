# 1.导包
import jieba
from base import setup_logger
logger=setup_logger("Preprocess")
# 2.封装一个preprocess_text函数，专门用于分本分词，返回结果一个列表
def preprocess_text(text):
    logger.info(f'开始分词：{text}')
    try:
        return jieba.lcut(text.lower())
    except AttributeError as e:
        logger.error(f'分词失败：{e}')
        return []
if __name__ == '__main__':
    print(preprocess_text('I am a boy'))
    print(preprocess_text('我爱北京天安门'))