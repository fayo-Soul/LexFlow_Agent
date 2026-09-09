# 1. 导包
# 递归加载所有文件信息
import os
import hashlib
import re
# 使用langchain_community包中的文档加载器(.txt、.md)
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
# langchain1.x版本分词器 => langchain_text_splitters => Markdown文本切割器
from langchain_text_splitters import MarkdownTextSplitter
# 引入datetime获取系统时间
from datetime import datetime
# 进度条
from tqdm import tqdm
# 加载自定义文本切割器
# 直接加载中文切分器，避免导入未使用的模型切分依赖。
from rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter import (
    ChineseRecursiveTextSplitter,
)
# 加载config.py与logger.py
from base import setup_logger, Config

# 2. 基础模块实例化
# 创建配置对象
conf = Config()
# 创建日志对象
logger = setup_logger("document_processor")

def _ocr_loader(loader_name, file_path):
    """仅在处理对应文件类型时加载 OCR 依赖。"""
    from rag_qa import edu_document_loaders
    loader_class = getattr(edu_document_loaders, loader_name)
    return loader_class(file_path)


# 3. 定义一个字典，用于保存支持的文件类型与对应的加载器
document_loaders = {
    # txt文件加载
    ".txt": TextLoader,
    # PDF文件加载
    ".pdf": lambda p: _ocr_loader("OCRPDFLoader", p),
    # Word
    ".doc": lambda p: _ocr_loader("OCRDOCLoader", p),
    ".docx": lambda p: _ocr_loader("OCRDOCLoader", p),
    # PPT
    ".ppt": lambda p: _ocr_loader("OCRPPTLoader", p),
    ".pptx": lambda p: _ocr_loader("OCRPPTLoader", p),
    # JPG、PNG图片
    ".jpg": lambda p: _ocr_loader("OCRIMGLoader", p),
    ".png": lambda p: _ocr_loader("OCRIMGLoader", p),
    # Markdown文件加载（使用TextLoader避免unstructured库初始化/兼容性问题）
    ".md": lambda p: TextLoader(p, encoding="utf-8")
}

# 法律条文目录 -> 5 大法律领域映射
LAW_SOURCE_MAP = {
    # 刑法
    "刑法": "刑法",
    # 民法
    "民法典": "民法",
    "民法商法": "民法",
    # 行政法
    "行政法": "行政法",
    "行政法规": "行政法",
    # 劳动法（目前数据集中在社会法，暂按其他法处理；后续可细化）
    # 其他法
    "社会法": "其他法",
    "经济法": "其他法",
    "宪法": "其他法",
    "宪法相关法": "其他法",
    "司法解释": "其他法",
    "案例": "其他法",
    "DLC": "其他法",
    "其他": "其他法",
    "部门规章": "其他法",
    "诉讼与非诉讼程序法": "其他法",
}


# 4. 定义函数，用于从文件夹中加载所有文档
def load_documents_from_directory(directory_path):
    documents = []  # 创建一个空列表，用于保存所有文档
    # 获取所有支持的数据格式
    supported_extensions = document_loaders.keys()
    # 定义一个变量source,用于记录对应的学科/法律领域名称
    # 优先从 "法律条文/刑法" 这类路径中提取领域名，否则回退到目录名
    normalized_path = directory_path.replace("\\", "/")
    if "法律条文/" in normalized_path:
        parts = normalized_path.split("法律条文/")[-1].split("/")
        source = parts[0] if parts[0] else os.path.basename(directory_path)
    else:
        source = os.path.basename(directory_path).replace("_data", "")

    # 统一映射到 5 大法律领域
    source = LAW_SOURCE_MAP.get(source, source)
    # 递归获取指定目录下的所有文件
    # root代表文件所在目录，_代表目录下的所有目录，files代表目录下的所有文件，也包含子目录下的文件
    # _在项目开发中，往往代表占位符，这个变量不需要使用，直接忽略
    for root, _, files in os.walk(directory_path):
        # 遍历所有文件
        for file in files:
            # 拼接文件绝对路径
            file_path = os.path.join(root, file).replace("\\", "/")
            file_extension = os.path.splitext(file)[1].lower()
            # 判断文件类型，然后使用对应的加载器加载文件 => .txt
            if file_extension in supported_extensions:
                try:
                    # 指定加载器类
                    loader_class = document_loaders[file_extension]
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    # 调用loader方法加载文档内容
                    loaded_docs = loader.load()  # [Document(), Document()]
                    title = os.path.splitext(file)[0]
                    document_id = hashlib.sha256(
                        file_path.encode("utf-8")
                    ).hexdigest()[:24]
                    if "司法解释" in normalized_path:
                        document_type = "judicial_interpretation"
                    elif "案例" in normalized_path:
                        document_type = "case"
                    else:
                        document_type = "regulation"
                    for doc in loaded_docs:
                        # 添加元数据，学科信息
                        doc.metadata['source'] = source
                        # 添加元数据，文件路径
                        doc.metadata['file_path'] = file_path
                        # 添加元数据，创建时间
                        doc.metadata['timestamp'] = datetime.now().isoformat()
                        doc.metadata['document_id'] = document_id
                        doc.metadata['title'] = title
                        doc.metadata['law_name'] = title
                        doc.metadata['document_type'] = document_type
                        doc.metadata['effective_status'] = "unknown"
                    # 使用列表.extend()方法，负责将两个列表进行合并[] + []
                    documents.extend(loaded_docs)
                    # 打印日志
                    logger.info(f"成功加载：{file_path}文件")
                except Exception as e:
                    logger.error(f"加载{file_path}文件时出错：{e}")
            else:
                # 打印日志，弹出提示，不支持的文件类型
                logger.warning(f"不支持的文件类型：{file_path}")
    # 所有逻辑代码编写完成后，返回所有文档
    return documents

# 5. 定义函数，process_documents用于处理所有文档（切割文档）
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE, child_chunk_size=conf.CHILD_CHUNK_SIZE, chunk_overlap=conf.CHUNK_OVERLAP):
    # 加载指定路径下的所有文档 => [Document(metadata/page_content), Document()]
    documents = load_documents_from_directory(directory_path)
    # 初始化父块与子块切割器（中文文本切割）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # markdown切割器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化空列表，用于保存所有子块信息
    child_chunks = []
    # 遍历所有文档，i代表第几个文档，doc代表文档内容
    for i, doc in enumerate(tqdm(documents, desc="处理法律条文文档", unit="doc")):
        # 获取文件对应的后缀 => metadata['file_path'],splitext,[1].lower()
        file_extension = os.path.splitext(doc.metadata['file_path'])[1].lower()
        # print(os.path.splitext(doc.metadata['file_path']))
        # ('E:/tools/PyCharm 2026.1/uv-project/rag_qa/data/ai_data/人工智能就业课课程大纲(测试)', '.md')
        # 定义一个变量，用于判断文档是否为.md文件
        is_markdown = (file_extension == ".md")
        # 根据文档类型，选择对应的切割器 => markdown => markdown，反之则切割中文文本
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(f"处理文档：{doc.metadata['file_path']}, 使用的切割器：{'Markdown' if is_markdown else 'ChineseRecursive'}")

        # 使用父块切割器对文档进行切割
        parent_docs = parent_splitter_to_use.split_documents([doc])
        # 对父块进行切割，获取所有子块
        # parent_doc => metadata => ['parent_id'.'parent_content']
        for j, parent_doc in enumerate(parent_docs):
            # print(f'doc_{i}_parent_{j}')
            # doc_0_parent_0
            # doc_0_parent_1
            parent_id=f'doc_{i}_parent_{j}'
            parent_doc.metadata['parent_id']=parent_id
            parent_doc.metadata['parent_content']=parent_doc.page_content

            # 使用子块切割器对父块进行切割
            sub_chunks=child_splitter_to_use.split_documents([parent_doc])
            for k,sub_chunk in enumerate(tqdm(sub_chunks, desc="切分子块", leave=False, unit="chunk")):
                # print(parent_docs)
                # print('-' * 40)
                # print(sub_chunk)
                # print('*' * 40)
                # 给子块增加元数据、父块id、父块内容、子块id => 后期检索问题答案时，先获取子块id，再根据父块id获取父块内容
                sub_chunk.metadata['parent_id']=parent_id
                sub_chunk.metadata['parent_content']=parent_doc.page_content
                sub_chunk.metadata['id']=f'{parent_id}_child_{k}'
                sub_chunk.metadata['chunk_id'] = (
                    f"{sub_chunk.metadata.get('document_id', 'document')}-"
                    f"{parent_id}-child-{k}"
                )
                article_match = re.search(
                    r"第[零〇一二三四五六七八九十百千万两0-9]+条"
                    r"(?:之[零〇一二三四五六七八九十百千万两0-9]+)?",
                    sub_chunk.page_content,
                )
                if article_match:
                    sub_chunk.metadata['article_number'] = article_match.group(0)
                    sub_chunk.metadata['location'] = article_match.group(0)
                # 将子块添加到列表中
                child_chunks.append(sub_chunk)
            # 当父块循环完毕，打印日志
            logger.info(f'子块数量：{len(child_chunks)}')
    return child_chunks



if __name__ == '__main__':
    process_documents("E:/tools/PyCharm 2026.1/uv-project/rag_qa/data/ai_data")
