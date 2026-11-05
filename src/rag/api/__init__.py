"""HTTP 层（O3）：FastAPI app 工厂 + 请求/响应契约。

    uvicorn src.rag.api.app:app --port 8000
    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/api/v1/ask -H "Content-Type: application/json" -d '{"question":"..."}'

子模块按需导入（`from src.rag.api.app import create_app`），避免 import 包时就把
FastAPI 与重量级依赖全拉进来。
"""
