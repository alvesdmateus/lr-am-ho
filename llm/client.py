import json
import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a single function/tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


class LLMResponse(BaseModel):
    """Parsed response from the LLM."""
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = Field(default_factory=dict)


class LLMClient:
    """
    Unified LLM Client for OpenAI-compatible APIs.
    Supports: Ollama, Google Gemini, Qwen/DashScope, vLLM, or any compatible endpoint.

    Provider setup examples:
        Ollama:   LLMClient(base_url="http://localhost:11434/v1", api_key="ollama", model="qwen2.5-coder")
        Gemini:   LLMClient(base_url="https://generativelanguage.googleapis.com/v1beta/openai", api_key="GEMINI_KEY", model="gemini-2.5-flash")
        Qwen:     LLMClient(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", api_key="QWEN_KEY", model="qwen-max")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        model: str = "qwen2.5-coder",
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """
        Sends a chat completion request and returns a parsed LLMResponse.
        Supports multi-turn messages, tool/function calling, and JSON mode.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                return self._parse_response(data)
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.max_retries:
                    continue
                raise ConnectionError(
                    f"LLM request failed after {self.max_retries + 1} attempts: {e}"
                ) from e

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Parses raw API JSON into a structured LLMResponse."""
        choice = data["choices"][0]
        message = choice["message"]

        tool_calls: List[ToolCall] = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                args_raw = tc["function"]["arguments"]
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                else:
                    args = args_raw

                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=tc["function"]["name"],
                        arguments=args,
                    )
                )

        usage = data.get("usage", {})

        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )

    def close(self) -> None:
        """Closes the underlying HTTP client."""
        self.client.close()
