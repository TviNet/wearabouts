from agent.models import LlmMessage, LlmMessageContentItem, TextItem
from agent.tools import JupyterCodeActionParser
from pydantic import BaseModel
from typing import List


class JupyterCodeAgentPrompt(BaseModel):
    FIXED_SYSTEM_PROMPT: str = f"""
Background:
    - You are a data analysis agent.
    - You are given a jupyter notebook and a task.
    - A jupyter notebook is a stateful collection of cells which are executed in order.

Goal:
    - Given the state of the jupyter notebook, and the task, in this turn, what is the next action to modify the notebook you should take?
    - The following are the actions you can take, strictly follow the format:
{JupyterCodeActionParser.get_actions_response_template()}

    - Note: You may use multiple actions in a single turn.

Strategies for generating actions:
    Planning:
    - Breaking down the task into smaller independent sub-tasks and listing them in a numbered list.
    - Use comments / markdown cells to explain your thought process and reason before implementing.
    
    Coding:
    - Try to limit yourself to around 20 lines of code in a single cell.
    - For this turn, try to write simple small blocks of code that you can validate. You can always add more cells in the future.
    - Do not repeat code from previous cells, unless it is necessary.

    Data types:
    - You can use print statements to inspect / understand the data / debug the code.
    - If the data type is a Dict, or List, first use print statements to inspect / understand the keys and values.
    - Check the final answer for task completion before stopping.
"""
    ADDITIONAL_SYSTEM_PROMPT: str
    NOTEBOOK_STATE_PREAMBLE: str = """
The following is the current state of the notebook:
```
"""
    NOTEBOOK_STATE_POSTAMBLE: str = """
```
"""

    def get_task_statement(self, task: str) -> str:
        return f"""
The following is the task we need to complete:
{task}
"""

    def get_notebook_state_content(
        self, notebook_state: List[LlmMessageContentItem]
    ) -> List[LlmMessageContentItem]:
        return (
            [TextItem(type="text", text=self.NOTEBOOK_STATE_PREAMBLE)]
            + notebook_state
            + [TextItem(type="text", text=self.NOTEBOOK_STATE_POSTAMBLE)]
        )

    def forward(
        self,
        task: str,
        notebook_state: List[LlmMessageContentItem],
    ) -> List[LlmMessage]:
        llm_messages = [
            LlmMessage(
                role="system",
                content=[
                    TextItem(type="text", text=self.FIXED_SYSTEM_PROMPT),
                    TextItem(type="text", text=self.ADDITIONAL_SYSTEM_PROMPT),
                ],
            )
        ]
        user_message_content: List[LlmMessageContentItem] = [
            TextItem(type="text", text=self.get_task_statement(task))
        ]
        notebook_state_content = self.get_notebook_state_content(notebook_state)
        user_message_content.extend(notebook_state_content)
        llm_messages.append(
            LlmMessage(
                role="user",
                content=user_message_content,
            )
        )

        return llm_messages
