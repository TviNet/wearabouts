import logging

from agent.models import (
    CellOutputTypes,
    ImageItem,
    LlmMessageContentItem,
    TextItem,
)
from app.constants import MAX_CELL_OUTPUT_LENGTH
from nbformat import NotebookNode
from pydantic import BaseModel
from sandbox.notebook_sandbox import CellType, JupyterSandbox
from typing import Any, ClassVar, Dict, List, Tuple
from utils.parsing import (
    extract_block_from_tags,
    extract_blocks_from_tags,
    try_to_parse_as_int,
)

logger = logging.getLogger(__name__)


class AddCellAction(BaseModel):
    name: ClassVar[str] = "add_cell"
    type: ClassVar[str] = "type"
    idx: ClassVar[str] = "idx"
    content: ClassVar[str] = "content"

    @staticmethod
    def get_response_template() -> str:
        return f"""
<{AddCellAction.name}>
<{AddCellAction.type}>Cell type i.e. either `code` or `markdown`. Type: enum</{AddCellAction.type}>
<{AddCellAction.idx}>Cell index i.e. the index of the cell to add to. Type: int</{AddCellAction.idx}>
<{AddCellAction.content}>
Cell content i.e. the content of the cell to add. Type: str
</{AddCellAction.content}>
</{AddCellAction.name}>
"""

    @staticmethod
    def handle_add_action(
        response: str, sandbox: JupyterSandbox, notebook: NotebookNode
    ) -> NotebookNode:
        add_entries = extract_blocks_from_tags(response, AddCellAction.name)
        for add_entry in add_entries:
            add_type = extract_block_from_tags(add_entry, AddCellAction.type)
            add_content = extract_block_from_tags(add_entry, AddCellAction.content)
            add_idx_str = extract_block_from_tags(add_entry, AddCellAction.idx)
            add_idx = try_to_parse_as_int(add_idx_str)
            if not add_type or not add_content or add_idx is None:
                continue
            notebook = sandbox.add_cell(
                notebook, add_content, cell_type=CellType(add_type), idx=add_idx
            )
        return notebook


class ModifyCellAction(BaseModel):
    name: ClassVar[str] = "modify_cell"
    type: ClassVar[str] = "type"
    idx: ClassVar[str] = "idx"
    content: ClassVar[str] = "content"

    @staticmethod
    def get_response_template() -> str:
        return f"""
<{ModifyCellAction.name}>
<{ModifyCellAction.type}>Cell type i.e. either `code` or `markdown`. Type: enum</{ModifyCellAction.type}>
<{ModifyCellAction.idx}>Cell index i.e. the index of the cell to modify. Type: int</{ModifyCellAction.idx}>
<{ModifyCellAction.content}>
Cell content i.e. the content of the cell to modify. Type: str
</{ModifyCellAction.content}>
</{ModifyCellAction.name}>
"""

    @staticmethod
    def handle_modify_action(
        response: str, sandbox: JupyterSandbox, notebook: NotebookNode
    ) -> NotebookNode:
        modify_entries = extract_blocks_from_tags(response, ModifyCellAction.name)
        for modify_entry in modify_entries:
            modify_idx_str = extract_block_from_tags(modify_entry, ModifyCellAction.idx)
            modify_idx = try_to_parse_as_int(modify_idx_str)
            modify_content = extract_block_from_tags(
                modify_entry, ModifyCellAction.content
            )
            if modify_idx is None or not modify_content:
                continue
            notebook = sandbox.modify_cell(notebook, modify_idx, modify_content)
        return notebook


class DeleteCellAction(BaseModel):
    name: ClassVar[str] = "delete_cell"
    idx: ClassVar[str] = "idx"

    @staticmethod
    def get_response_template() -> str:
        return f"""
<{DeleteCellAction.name}>
<{DeleteCellAction.idx}>Cell index i.e. the index of the cell to delete. Type: int</{DeleteCellAction.idx}>
</{DeleteCellAction.name}>
"""

    @staticmethod
    def handle_delete_action(
        response: str, sandbox: JupyterSandbox, notebook: NotebookNode
    ) -> NotebookNode:
        delete_entries = extract_blocks_from_tags(response, DeleteCellAction.name)
        for delete_entry in delete_entries:
            delete_idx_str = extract_block_from_tags(delete_entry, DeleteCellAction.idx)
            delete_idx = try_to_parse_as_int(delete_idx_str)
            if delete_idx is None:
                continue
            notebook = sandbox.delete_cell(notebook, delete_idx)
        return notebook


class StopAction(BaseModel):
    name: ClassVar[str] = "stop"

    @staticmethod
    def get_response_template() -> str:
        return f"""
<{StopAction.name}>
</{StopAction.name}>
"""

    @staticmethod
    def handle_stop_action(response: str) -> bool:
        stop_entries = extract_blocks_from_tags(response, StopAction.name)
        if stop_entries:
            return True
        return False


class JupyterCodeActionParser:
    @staticmethod
    def get_actions_response_template() -> str:
        return f"""
Possible actions:
1. Add a cell
{AddCellAction.get_response_template()}
2. Modify a cell
{ModifyCellAction.get_response_template()}
3. Delete a cell
{DeleteCellAction.get_response_template()}
4. Stop
{StopAction.get_response_template()}
"""

    @staticmethod
    def response_to_actions(
        response: str, sandbox: JupyterSandbox, notebook: NotebookNode
    ) -> Tuple[NotebookNode, bool]:
        print()
        notebook = AddCellAction.handle_add_action(response, sandbox, notebook)
        notebook = ModifyCellAction.handle_modify_action(response, sandbox, notebook)
        notebook = DeleteCellAction.handle_delete_action(response, sandbox, notebook)
        should_stop = StopAction.handle_stop_action(response)
        return notebook, should_stop


class JupyterCodeParser:
    @staticmethod
    def convert_output_to_string(output: Dict[str, Any]) -> LlmMessageContentItem:
        if output.get("output_type", "") == CellOutputTypes.STREAM.value:
            return TextItem(type="text", text=output.get("text", ""))
        elif output.get("output_type", "") == CellOutputTypes.ERROR.value:
            return TextItem(
                type="text",
                text=f"""    
{output.get("ename")}: {output.get("evalue")}
{output.get("traceback")}
""",
            )
        elif output.get("output_type", "") == CellOutputTypes.DISPLAY_DATA.value:
            if output.get("data", {}).get("image/png", None):
                image_data = output.get("data", {}).get("image/png", None)
                return ImageItem(
                    type="image_url",
                    image_url={"url": f"data:image/png;base64,{image_data}"},
                )
        return TextItem(type="text", text="")

    @staticmethod
    def render_notebook(
        notebook: NotebookNode, include_outputs=True
    ) -> List[LlmMessageContentItem]:
        state: List[LlmMessageContentItem] = []
        for idx, cell in enumerate(notebook.cells):
            content = cell.source
            if CellType(cell.cell_type) == CellType.MARKDOWN:
                state.append(
                    TextItem(
                        type="text",
                        text=f"""# <cell {idx}>\n{content}\n# </cell {idx}>""",
                    )
                )

            elif CellType(cell.cell_type) == CellType.CODE:
                state.append(
                    TextItem(
                        type="text",
                        text=f"""# <cell {idx}: input>\n{content}\n# </cell {idx}: input>""",
                    )
                )
                if include_outputs:
                    outputs = cell.outputs if hasattr(cell, "outputs") else []
                    output_repr = []
                    total_output_length = 0
                    for output in outputs:
                        current_output_repr = (
                            JupyterCodeParser.convert_output_to_string(output)
                        )
                        if current_output_repr["type"] == "text":
                            # skip empty text outputs
                            if current_output_repr["text"] == "":
                                continue
                            if total_output_length > MAX_CELL_OUTPUT_LENGTH:
                                continue
                            if total_output_length + len(
                                current_output_repr["text"]
                            ) > (MAX_CELL_OUTPUT_LENGTH):
                                available_space = (
                                    MAX_CELL_OUTPUT_LENGTH - total_output_length
                                )
                                current_output_repr["text"] = (
                                    current_output_repr["text"][:available_space]
                                    + "... (truncated)"
                                )
                            total_output_length += len(current_output_repr["text"])
                        output_repr.append(current_output_repr)
                    if output_repr:
                        state += (
                            [
                                TextItem(
                                    type="text",
                                    text=f"""\n# <cell {idx}: output>\n""",
                                )
                            ]
                            + output_repr
                            + [
                                TextItem(
                                    type="text",
                                    text=f"""\n# </cell {idx}: output>\n""",
                                )
                            ]
                        )
            else:
                logger.error(f"Invalid cell type: {cell.cell_type}")
        # wrap in python code block
        state = (
            [TextItem(type="text", text="```python\n")]
            + state
            + [TextItem(type="text", text="```")]
        )
        return state


if __name__ == "__main__":
    sandbox = JupyterSandbox()
    # test jupyter code execution tool
    notebook = sandbox.create_notebook()
    actions = """
<add_cell>
<type>code</type>
<content>
print("Hello, world!")
</content>
<idx>0</idx>
</add_cell>
<add_cell>
<type>code</type>
<content>
print("Hello, world! 2!")
</content>
<idx>  -1</idx>
</add_cell>
<modify_cell>
<type> code</type>
<idx>0</idx>
<content>
print("Hello, world! 3!")
</content>
</modify_cell>
<delete_cell>
<idx>1</idx>
</delete_cell>
<stop></stop>
"""

    notebook, should_stop = JupyterCodeActionParser.response_to_actions(
        actions, sandbox, notebook
    )
    import json

    print(json.dumps(JupyterCodeParser.render_notebook(notebook), indent=4))
    print(should_stop)
