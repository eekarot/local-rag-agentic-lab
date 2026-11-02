"""摄取层异常。

统一基类让调用方可以一次捕获，同时保留细分类别有可执行的错误信息。
"""


class IngestionError(Exception):
    """摄取层所有错误的基类。"""


class UnsupportedFormatError(IngestionError):
    """没有解析器认领该文件后缀。错误信息里必须列出已注册后缀。"""


class TokenizerUnavailable(IngestionError):
    """本地 tokenizer 文件不存在或无法加载。错误信息里必须给出获取方式。"""
