# LexFlow 本地前端

前端通过本机安全代理访问虚拟机中的 LexFlow Agent API。API Token 仅由代理进程从虚拟机读取，不会发送到浏览器。

## 启动

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\frontend\start.ps1
```

然后访问：

```text
http://127.0.0.1:3000
```

默认连接：

- Agent API：`http://192.168.88.100:8000`
- SSH：`root@192.168.88.100`
- SSH 私钥：`%USERPROFILE%\.ssh\lexflow_vm`

可通过环境变量 `LEXFLOW_FRONTEND_HOST`、`LEXFLOW_FRONTEND_PORT`、`LEXFLOW_API_UPSTREAM`、`LEXFLOW_VM_HOST`、`LEXFLOW_VM_USER` 和 `LEXFLOW_SSH_KEY` 覆盖默认配置。
