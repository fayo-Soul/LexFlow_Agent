"""案例稀疏向量的稳定词项映射。"""

import hashlib


def stable_token_id(token: str, dimensions: int = 250000) -> int:
    digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions
