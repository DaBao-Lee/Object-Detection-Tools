"""视觉大模型（OpenAI 兼容接口）调用封装。

OpenAPI        面向 OpenAI 兼容 /v1/chat/completions 的流式对话客户端
getSystemPrompt  读取 system prompt 文本文件
chat_stream    一次性拿到完整回复文本的函数式入口

依赖 `requests` 为**惰性导入**：仅在真正发请求时才 import，因此
`from ObjectDetection import OpenAPI` 在未装 requests 的环境也能成功（只是调用时会报错）。

典型用法（本地 llama.cpp 服务）：

    from ObjectDetection import OpenAPI, getSystemPrompt

    agent = OpenAPI(
        "http://127.0.0.1:8080/v1/chat/completions",
        "",                     # api key，本地服务一般留空
        "Qwen2.5-VL-7B",       # 模型名
        "LLAMA.CPP",           # provider 标识（仅作记录）
    )
    gen = agent.openAPIRequest(
        "描述这张图片里的病害",
        image_prompt="data:image/jpeg;base64,...",
        systemPrompt=getSystemPrompt("./systemPrompt.txt"),
    )
    print(agent.parseGenerator(gen))
"""

import json
from typing import Generator, Optional

# 默认 system prompt（未显式传入时使用）
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def getSystemPrompt(path: str) -> str:
    """读取 system prompt 文件的全文。

    path : prompt 文件路径（UTF-8 文本）
    返回 : 文件内容（保留原始换行）
    """

    with open(path, encoding="utf-8", mode="r") as f:
        return "\n".join(f.readlines())


class OpenAPI:
    """OpenAI 兼容接口的流式对话客户端（支持多模态图文输入）。

    四个构造参数均为调用方必填，不做"留空即推导"的隐式行为：

    specific_api_url : 完整接口地址，如 http://127.0.0.1:8080/v1/chat/completions
    specific_api_key : API Key；本地 llama.cpp 等服务一般留空字符串
    model            : 模型名（写入请求体的 model 字段）
    provider         : 服务提供方标识（写入请求体的 provider 字段）
    """

    def __init__(
        self,
        specific_api_url: str,
        specific_api_key: str,
        model: str,
        provider: str,
    ) -> None:

        self.specific_api_url = specific_api_url
        self.specific_api_key = specific_api_key
        self.model = model
        self.provider = provider

    def headerCreate(self) -> dict:
        """构造请求头。api_key 非空时才附带 Authorization 字段。"""

        headers = {"Content-Type": "application/json"}

        if self.specific_api_key:
            headers["Authorization"] = f"Bearer {self.specific_api_key}"

        return headers

    def playLoadCreate(
        self,
        prompt: str,
        image_prompt: Optional[str] = None,
        temperature: float = 0.9,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        think: bool = False,
        systemPrompt: str = "",
    ) -> dict:
        """组装请求体（payload）。

        image_prompt 传 base64 data-url 时走多模态分支（图文混合输入），
        否则走纯文本分支。
        """

        if image_prompt is None:
            payload = {
                "messages": [
                    {"role": "system", "content": systemPrompt},
                    {"role": "user", "content": prompt},
                ],
            }
        else:
            payload = {
                "messages": [
                    {"role": "system", "content": systemPrompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_prompt}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            }

        payload.update(
            {
                "model": self.model,
                "provider": self.provider,
                "top_p": top_p,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": think},
            }
        )

        return payload

    def openAPIRequest(
        self,
        prompt: str,
        image_prompt: Optional[str] = None,
        temperature: float = 0.9,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        think: bool = False,
        systemPrompt: Optional[str] = None,
    ) -> Generator:
        """发起流式请求，逐块 yield 文本增量。

        这是个生成器：真正发出网络请求发生在第一次迭代时（惰性）。
        systemPrompt 为 None 时使用 DEFAULT_SYSTEM_PROMPT。

        依赖 `requests`（惰性导入，未安装时抛 ImportError）。
        """

        if systemPrompt is None:
            systemPrompt = DEFAULT_SYSTEM_PROMPT

        try:
            import requests
        except ImportError as exc:  # 惰性导入，未安装 requests 时给出明确提示
            raise ImportError(
                "调用 openAPIRequest 需要 requests，请执行：pip install requests"
            ) from exc

        headers = self.headerCreate()
        playload = self.playLoadCreate(
            prompt,
            image_prompt,
            temperature,
            top_p,
            max_tokens,
            think,
            systemPrompt,
        )

        response = requests.post(
            self.specific_api_url, headers=headers, json=playload, stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue

            data = line[6:]  # 去掉 'data: ' 前缀
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                yield content

    def parseGenerator(self, generator: Generator, show: bool = True) -> str:
        """把流式生成器收成完整字符串。

        show=True 时同时逐块打印（用于交互式调试）。
        返回拼接后的完整文本。
        """

        response = ""
        for chunk in generator:
            if show:
                print(chunk, end="", flush=True)
            response += chunk

        return response


def chat_stream(
    api_url: str,
    api_key: str,
    model: str,
    provider: str,
    prompt: str,
    image_prompt: Optional[str] = None,
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    think: bool = False,
    systemPrompt: Optional[str] = None,
    show: bool = False,
) -> str:
    """函数式入口：一次调用拿到完整回复文本。

    内部复用 OpenAPI 类，适合不关心流式过程、只要最终字符串的场景
    （如后端接口层生成报告）。

    返回 : 模型回复的完整文本
    """

    agent = OpenAPI(api_url, api_key, model, provider)
    generator = agent.openAPIRequest(
        prompt,
        image_prompt=image_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        think=think,
        systemPrompt=systemPrompt,
    )
    return agent.parseGenerator(generator, show=show)
