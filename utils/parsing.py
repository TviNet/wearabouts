import re

from typing import List, Optional


def try_to_parse_as_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def extract_codeblock_from_backticks(
    raw_response: str, codeblock_name: str
) -> List[str]:
    # find all codeblocks with the given name
    # remove the backticks and the codeblock name
    # return the codeblocks
    codeblocks = re.findall(f"```{codeblock_name}\n(.*?)\n```", raw_response, re.DOTALL)
    # remove the backticks and the codeblock name
    codeblocks = [
        codeblock.replace(f"```{codeblock_name}\n", "").replace("\n```", "")
        for codeblock in codeblocks
    ]

    return codeblocks


def extract_blocks_from_tags(raw_response: str, tag: str) -> List[str]:
    # find all blocks with the given tag
    # remove the tag
    # return the blocks
    blocks = re.findall(f"<{tag}>(.*?)</{tag}>", raw_response, re.DOTALL)
    # remove the tag
    blocks = [
        block.replace(f"<{tag}>", "").replace(f"</{tag}>", "") for block in blocks
    ]
    return blocks


def extract_block_from_tags(raw_response: str, tag: str) -> Optional[str]:
    blocks = extract_blocks_from_tags(raw_response, tag)
    return blocks[0] if blocks else None
