r"""
TODO 温馨提示：
如果报错：zipfile.BadZipFile: File is not a zip file
则需要拷贝资料下的nltk_data到{项目的虚拟环境}/share目录下。
比如：我的项目环境是rag。目录是：C:\Programming\Anaconda3\envs\rag
则我需要拷贝到：C:\Programming\Anaconda3\envs\rag\share目录下。
"""

# TODO 1.官网链接
# https://reference.langchain.com/python/integrations/langchain_unstructured/

from langchain_community.document_loaders import UnstructuredMarkdownLoader
# 准备文件
mk_file_path = "D:/pythonProject/integrated_qa_system/rag_qa/data/ai_data/人工智能就业课课程大纲(测试).md"
# 创建加载器对象
loader = UnstructuredMarkdownLoader(file_path=mk_file_path)
print("loader-->", loader)
docs = []
# 加载文件
docs_lazy = loader.lazy_load()
# 遍历
for doc in docs_lazy:
    # print('doc-->', doc)
    docs.append(doc)
print('docs[0].content-->', docs[0].page_content)
print('docs[0].metadata-->', docs[0].metadata)
