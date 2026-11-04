"""语言识别与后续的语言路由（D22 / D29）。

子模块按需导入：`from src.rag.lang.detect import detect`。
这里不做 eager import，否则 `python -m src.rag.lang.detect` 会触发
「found in sys.modules prior to execution」的 RuntimeWarning。
"""
