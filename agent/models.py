from enum import Enum
from pydantic import BaseModel
from typing import List, Literal, TypeAlias, TypedDict, Union


# models enum
class LlmModel(Enum):
    # GPT_4O = "openai/gpt-4o"
    # SONNET_3_5 = "anthropic/claude-3-5-sonnet-20240620"

    GEMINI_2_0_FLASH = "gemini/gemini-2.0-flash"
    GEMINI_1_5_FLASH_8B = "gemini/gemini-1.5-flash-8b"


class ImageItem(TypedDict):
    type: Literal["image"]
    image_url: str


class TextItem(TypedDict):
    type: Literal["text"]
    text: str


LlmMessageContentItem: TypeAlias = Union[ImageItem, TextItem]


class LlmMessage(BaseModel):
    role: str
    content: List[LlmMessageContentItem]


class LlmProviderConfig(BaseModel):
    model: LlmModel
    api_key: str


class LlmParameterConfig(BaseModel):
    temperature: float = 0.1
    max_tokens: int = 4096
