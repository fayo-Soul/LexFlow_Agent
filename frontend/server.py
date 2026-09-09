"""Local-only frontend server and authenticated proxy for LexFlow Agent."""

from __future__ import annotations

import json
import mimetypes
import subprocess
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "127.0.0.1"
PORT = 3000
UPSTREAM = "http://192.168.88.100:8000"
ROOT = Path(__file__).resolve().parent
SSH_KEY = Path.home() / ".ssh" / "lexflow_vm"


def load_api_token() -> str:
    command = [
        "ssh", "-i", str(SSH_KEY), "-o", "BatchMode=yes",
        "root@192.168.88.100", "cat /opt/lexflow-agent/.api-token",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    token = result.stdout.strip()
    if not token:
        raise RuntimeError("虚拟机 API Token 为空")
    return token


API_TOKEN = load_api_token()


class Handler(BaseHTTPRequestHandler):
    server_version = "LexFlowFrontend/1.0"

    def do_GET(self):
        if self.path.startswith("/api/"):
            self.proxy()
            return
        path = self.path.split("?", 1)[0]
        target = ROOT / ("index.html" if path == "/" else path.lstrip("/"))
        try:
            target = target.resolve()
            target.relative_to(ROOT)
        except (ValueError, OSError):
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        self.proxy()

    def do_DELETE(self):
        self.proxy()

    def proxy(self):
        upstream_path = self.path.removeprefix("/api")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None
        request = urllib.request.Request(
            UPSTREAM + upstream_path,
            data=body,
            method=self.command,
            headers={
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Accept": self.headers.get("Accept", "application/json"),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=1900) as response:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while chunk := response.read(8192):
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", error.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as error:
            payload = json.dumps({"detail": f"代理请求失败: {error}"}, ensure_ascii=False).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[frontend] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    print(f"LexFlow frontend: http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
