# 把 Agent 打包成容器。
#
# 重點：這個映像檔裡**沒有 Ollama**。
# 因為我們用的是 direct API 模式（見核心路徑第 2 節），
# ollama.com 就是遠端的推論主機，容器只需要一個 HTTP client。
# 這也是為什麼映像檔可以這麼小——如果走本地代理模式，
# 你得把整個 Ollama runtime 和模型權重塞進來，那是好幾 GB。

FROM python:3.13-slim AS base

# 只裝服務真正需要的。langchain/langfuse/mcp 是教學用的選用套件，
# 生產映像檔沒必要背著它們。
RUN pip install --no-cache-dir "ollama>=0.6" "python-dotenv>=1.0"

# 用非 root 執行。容器逃逸的第一道防線，成本是零。
RUN useradd --create-home --uid 10001 agent
WORKDIR /app

COPY --chown=agent:agent examples/_client.py        examples/
COPY --chown=agent:agent examples/rag_common.py     examples/
COPY --chown=agent:agent examples/12_rag_cloud_only.py examples/
COPY --chown=agent:agent examples/18_deploy_server.py  examples/
COPY --chown=agent:agent examples/corpus/           examples/corpus/

USER agent

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    AGENT_TIMEOUT=120 \
    MAX_CONCURRENT=4

EXPOSE 8080

# API key 一定要在執行期注入，絕不 COPY 進映像檔也不寫在 ENV。
# 映像檔會被推到 registry、被別人 pull、每一層都留在歷史裡。
#   docker run -e OLLAMA_API_KEY=... 或用 secret manager

# 健康檢查打的是 /healthz，那個端點刻意不呼叫模型——
# 否則你的健康檢查會變成帳單。
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status==200 else 1)"

CMD ["python", "examples/18_deploy_server.py"]
