"""Download the two public BGE models required by Law RAG."""

from pathlib import Path

from modelscope import snapshot_download


target = Path("/models")
target.mkdir(parents=True, exist_ok=True)

for repository, directory in (
    ("BAAI/bge-reranker-large", "bge-reranker-large"),
    ("BAAI/bge-m3", "bge-m3"),
):
    snapshot_download(repository, local_dir=str(target / directory))
