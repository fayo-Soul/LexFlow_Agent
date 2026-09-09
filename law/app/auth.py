"""API Token 身份与角色校验。"""

from dataclasses import dataclass
from hmac import compare_digest


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str


def authenticate_bearer(
    authorization: str | None,
    admin_token: str,
    researcher_token: str = "",
) -> Principal | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    if admin_token and compare_digest(token, admin_token):
        return Principal(user_id="api-admin", role="admin")
    if researcher_token and compare_digest(token, researcher_token):
        return Principal(user_id="api-researcher", role="researcher")
    return None
