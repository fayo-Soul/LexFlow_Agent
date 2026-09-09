import json
import shlex

import paramiko


client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect(
    "192.168.88.100",
    username="root",
    key_filename=r"C:\Users\HP\.ssh\lexflow_vm",
    timeout=8,
)

_, stdout, _ = client.exec_command("docker inspect redis")
inspection = json.loads(stdout.read().decode())[0]

password = ""
for item in inspection["Config"].get("Env") or []:
    if item.startswith("REDIS_PASSWORD="):
        password = item.split("=", 1)[1]
        break

if not password:
    command = inspection["Config"].get("Cmd") or []
    for index, item in enumerate(command):
        if item == "--requirepass" and index + 1 < len(command):
            password = command[index + 1]
            break

if not password:
    raise RuntimeError("Redis password was not found in the container configuration")

env_path = "/opt/lexflow-agent/law/.env.deploy"
with client.open_sftp().file(env_path, "r") as env_file:
    lines = env_file.read().decode().splitlines()

updated = [
    f"REDIS_PASSWORD={password}" if line.startswith("REDIS_PASSWORD=") else line
    for line in lines
]
if not any(line.startswith("REDIS_PASSWORD=") for line in lines):
    updated.append(f"REDIS_PASSWORD={password}")

with client.open_sftp().file(env_path, "w") as env_file:
    env_file.write(("\n".join(updated) + "\n").encode())

client.exec_command(f"chmod 600 {shlex.quote(env_path)}")[1].channel.recv_exit_status()
client.close()
print("Redis credential synchronized without displaying it")
