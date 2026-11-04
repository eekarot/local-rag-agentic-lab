import requests
from .config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL

TIMEOUT = 300


def _post(path, payload):
    url = f"{OLLAMA_BASE_URL.rstrip('/')}{path}"
    env_var = "OLLAMA_EMBED_MODEL" if path.endswith("/embed") else "OLLAMA_CHAT_MODEL"
    model = payload.get("model", "?")
    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        # Ollama 的 404 几乎总是「模型不存在」，直接把服务器原文和可用模型摊开
        detail = ""
        try:
            detail = str(response.json().get("error") or "")
        except ValueError:
            detail = (response.text or "").strip()[:200]
        if response.status_code == 404:
            installed = _installed_models()
            raise RuntimeError(
                f"Ollama 上找不到模型 {model!r}：{detail or 'HTTP 404'}\n"
                f"当前已安装的模型：{installed or '（查询失败）'}\n"
                f"二选一：\n"
                f"  1) 拉取该模型：ollama pull {model}\n"
                f"  2) 换成已有模型：设置环境变量 {env_var}=<上面列表里的名字>"
            ) from exc
        raise RuntimeError(
            f"Ollama API 返回 HTTP {response.status_code}：{detail or '无响应体'}\n"
            f"请求地址：{url}，模型：{model!r}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"无法调用 Ollama API：{url}\n"
            "请确认 Ollama 正在运行，模型已下载，且端口为默认的 11434。\n"
            f"原始错误：{exc}"
        ) from exc


def _installed_models():
    """列出 Ollama 已安装的模型，用于把报错信息变成可执行的下一步。"""
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5
        )
        response.raise_for_status()
        return [m.get("name", "?") for m in response.json().get("models", [])]
    except (requests.RequestException, ValueError):
        return []


def embed_texts(texts):
    """调用 Ollama /api/embed，输入字符串列表，返回向量列表。"""
    if not texts:
        return []
    data = _post("/api/embed", {
        "model": OLLAMA_EMBED_MODEL,
        "input": texts,
    })
    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError(
            "Ollama 没有返回 embeddings。请检查 Embedding 模型名称，"
            "并确认 Ollama 版本支持 /api/embed。"
        )
    return embeddings


def chat_full(messages, temperature=0, seed=None, think=None, num_predict=None, fmt=None):
    """调用 /api/chat，返回内容与 Ollama 自带的 token 统计（日志 review 用）。

    `think=False` 用来关掉思考型模型（qwen3.5 等）的推理段：不关的话
    `message.content` 会是空串、真正的输出全在 `message.thinking` 里，
    表现为「调用不报错但答案是空的」，或者推理写太长直到 300 s 超时。
    """
    options = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if think is not None:
        payload["think"] = think
    if fmt is not None:
        payload["format"] = fmt
    data = _post("/api/chat", payload)
    message = data.get("message") or {}
    return {
        "content": message.get("content", ""),
        "thinking": message.get("thinking"),
        "model": data.get("model") or OLLAMA_CHAT_MODEL,
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "done_reason": data.get("done_reason"),
    }


def chat(messages, temperature=0, seed=None, think=None, num_predict=None, fmt=None):
    return chat_full(messages, temperature=temperature, seed=seed,
                     think=think, num_predict=num_predict, fmt=fmt)["content"]
