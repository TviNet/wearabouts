import litellm
import os

from agent.models import LlmMessage, LlmModel, LlmParameterConfig, LlmProviderConfig
from litellm import completion
from litellm.caching.caching import Cache, LiteLLMCacheType
from pydantic.dataclasses import dataclass
from typing import List

litellm.cache = Cache(type=LiteLLMCacheType.DISK)


@dataclass
class LlmClient:
    provider_config: LlmProviderConfig
    parameter_config: LlmParameterConfig

    def get_single_answer(self, messages: List[LlmMessage]) -> str:
        full_response = completion(
            messages=messages,
            model=self.provider_config.model.value,
            api_key=self.provider_config.api_key,
            temperature=self.parameter_config.temperature,
            max_tokens=self.parameter_config.max_tokens,
            caching=True,
        )
        return full_response["choices"][0]["message"]["content"]


def get_llm_client() -> LlmClient:
    return LlmClient(
        provider_config=LlmProviderConfig(
            model=LlmModel.GEMINI_2_0_FLASH, api_key=os.getenv("GEMINI_API_KEY")
        ),
        parameter_config=LlmParameterConfig(),
    )


if __name__ == "__main__":
    llm_client = get_llm_client()
    print(
        llm_client.get_single_answer(
            [LlmMessage(role="user", content="Hello, how are you?")]
        )
    )
