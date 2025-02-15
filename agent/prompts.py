from agent.models import LlmMessage, LlmMessageContentItem, TextItem
from agent.tools import JupyterCodeActionParser
from pydantic import BaseModel
from typing import List


class JupyterCodeAgentPrompt(BaseModel):
    FIXED_SYSTEM_PROMPT: str = f"""
You are a data analysis agent.
Given the state of the jupyter notebook, and the task, complete the task.

The following is the format of the actions you can take:
{JupyterCodeActionParser.get_actions_response_template()}

Strategies:
- Use comments / markdown cells to explain your thought process and reason before implementing.
- You can use print statements to inspect / understand the data / debug the code.
- If the data type is a Dict, first use print statements to inspect / understand the keys and values.
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
