# 常用指令。不確定要打什麼就先 `make`（等同 `make help`）。
#
# 設計原則：每個會「留下東西」的指令，都有一個對應的收拾指令。
# 我自己就忘記收過跑在背景的伺服器，所以 stop / clean 是一等公民。

SHELL      := /bin/bash
PY         ?= .venv/bin/python
IMAGE      ?= ollama-agent-demo
TAG        ?= local
PORT       ?= 8080
CONTAINER  ?= ollama-agent

.DEFAULT_GOAL := help
.PHONY: help setup check test eval eval-full docs-check _server_pids \
        serve stop docker-build docker-run docker-logs docker-stop docker-clean \
        status clean

## ---- 說明 ---------------------------------------------------------------

help:  ## 列出所有指令
	@echo "用法: make <target>"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "環境變數: PORT=$(PORT)  IMAGE=$(IMAGE):$(TAG)  CONTAINER=$(CONTAINER)"

## ---- 開發 ---------------------------------------------------------------

setup:  ## 建立 venv 並安裝依賴
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@echo "完成。接著跑 make check"

check:  ## 環境自檢（API key、可用模型、套件、本地 Ollama）
	$(PY) examples/00_check_setup.py

test:  ## 跑單元測試（0.2 秒，不呼叫模型）
	$(PY) examples/16_testing_agent.py

eval:  ## RAG 檢索指標（秒級，不呼叫模型）
	$(PY) examples/14_rag_eval.py

eval-full:  ## RAG 端到端評估（會呼叫雲端模型，花 GPU 時間）
	$(PY) examples/14_rag_eval.py --end-to-end

docs-check:  ## 文件一致性檢查（連結、錨點、範例引用）
	$(PY) check_docs.py

## ---- 本機跑服務 ---------------------------------------------------------

serve:  ## 在前景啟動 Agent HTTP 服務（Ctrl-C 停止）
	PORT=$(PORT) $(PY) examples/18_deploy_server.py

stop:  ## 停掉背景殘留的本機服務（不影響你自己的 ollama serve）
	@pids=$$($(MAKE) -s _server_pids); \
	if [ -z "$$pids" ]; then echo "沒有正在執行的本機服務"; else \
	  kill $$pids 2>/dev/null || true; \
	  for i in $$(seq 1 10); do \
	    [ -z "$$($(MAKE) -s _server_pids)" ] && break; sleep 1; \
	  done; \
	  left=$$($(MAKE) -s _server_pids); \
	  if [ -n "$$left" ]; then echo "優雅關閉逾時，強制結束"; kill -9 $$left; \
	  else echo "已優雅停止本機服務（PID $$pids）"; fi; \
	fi

# 用「誰在聽這個埠」找行程，再驗證 cmdline 確實是我們的服務。
# 不用 pgrep -f 比對字串——那會匹配到「命令列裡剛好含有這個檔名」的
# 呼叫端 shell，然後把自己殺掉。我實際踩過這個坑。
_server_pids:
	@(ss -tlnpH "sport = :$(PORT)" 2>/dev/null || ss -tlnp 2>/dev/null | grep ":$(PORT) ") \
	  | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u \
	  | while read -r pid; do \
	      tr '\0' ' ' < /proc/$$pid/cmdline 2>/dev/null \
	        | grep -q '18_deploy_server.py' && echo "$$pid"; \
	    done

## ---- Docker -------------------------------------------------------------

docker-build:  ## 建立映像檔
	docker build -t $(IMAGE):$(TAG) .
	@docker images $(IMAGE):$(TAG) --format "映像檔 {{.Repository}}:{{.Tag}}  {{.Size}}"

docker-run: docker-build  ## 建立並在背景啟動容器
	@test -n "$$OLLAMA_API_KEY" || { \
	  echo "缺少 OLLAMA_API_KEY。先 export，或 source 你的 .env"; exit 1; }
	@docker rm -f $(CONTAINER) >/dev/null 2>&1 || true
	docker run -d --name $(CONTAINER) -p $(PORT):8080 \
	  -e OLLAMA_API_KEY="$$OLLAMA_API_KEY" $(IMAGE):$(TAG)
	@sleep 3
	@echo "健康檢查: $$(curl -s localhost:$(PORT)/healthz)"
	@echo "就緒檢查: $$(curl -s localhost:$(PORT)/readyz)"
	@echo
	@echo "試一下:"
	@echo "  curl -X POST localhost:$(PORT)/ask -H 'Content-Type: application/json' \\"
	@echo "       -d '{\"question\":\"PR 太大要怎麼辦？\"}'"
	@echo "結束時: make docker-stop"

docker-logs:  ## 追容器日誌
	docker logs -f $(CONTAINER)

docker-stop:  ## 優雅停止並移除容器（保留映像檔）
	@docker stop -t 15 $(CONTAINER) >/dev/null 2>&1 && echo "已優雅停止" \
	  || echo "容器沒在跑"
	@docker rm -f $(CONTAINER) >/dev/null 2>&1 || true

docker-clean: docker-stop  ## 停止容器並刪除映像檔
	@docker rmi $(IMAGE):$(TAG) >/dev/null 2>&1 && echo "已刪除映像檔" \
	  || echo "沒有映像檔可刪"

## ---- 收拾 ---------------------------------------------------------------

status:  ## 看看目前有什麼還在跑
	@echo "── 本機服務 ──"
	@pids=$$($(MAKE) -s _server_pids); [ -n "$$pids" ] && ps -o pid,cmd --no-headers -p $$pids || echo "  （無）"
	@echo "── 佔用 $(PORT) 埠 ──"
	@(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) \
	  | grep ":$(PORT) " || echo "  （無）"
	@echo "── Docker 容器 ──"
	@docker ps -a --filter "name=$(CONTAINER)" \
	  --format "  {{.Names}}  {{.Status}}" 2>/dev/null | grep . || echo "  （無）"
	@echo "── 映像檔 ──"
	@docker images $(IMAGE) --format "  {{.Repository}}:{{.Tag}}  {{.Size}}" \
	  2>/dev/null | grep . || echo "  （無）"

clean: stop docker-clean  ## 全部收乾淨（本機服務 + 容器 + 映像檔 + 暫存）
	@rm -rf examples/__pycache__ examples/tools/__pycache__ __pycache__
	@rm -f examples/memory.db
	@echo "已清理暫存與長期記憶資料庫"
