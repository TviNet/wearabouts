from enum import Enum
from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Tuple, TypeAlias, TypedDict, Union


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


class CellOutputTypes(Enum):
    STREAM = "stream"
    ERROR = "error"
    DISPLAY_DATA = "display_data"
    EXECUTE_RESULT = "execute_result"
    OTHER = "other"


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"


class StreamOutput(TypedDict):
    output_type: str
    name: str
    text: str


class ErrorOutput(TypedDict):
    output_type: str
    ename: str
    evalue: str
    traceback: List[str]


class DisplayDataOutput(TypedDict):
    output_type: str
    data: Dict[str, Any]


class ExecuteResultOutput(TypedDict):
    output_type: str
    data: Dict[str, Any]


StateItem: TypeAlias = Tuple[str, str]
