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
    test_img = "iVBORw0KGgoAAAANSUhEUgAAAh8AAAGzCAYAAACPa3XZAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAIz9JREFUeJzt3QuUVeV9N+AXIVxEGAVvUECIsTUGg0bFqKmBlEosIZIVTTSmQUzUJqhFkyhjo4Z4AaM1rCoFTVs0rahJGtRqo7V4oU0gCkhb04oSxRBdiladEawjgfOt/+53pjPDgIDnvMOZeZ61tuPZZ8/e774w+3feyz7dSqVSKQEAZLJbrg0BAAThAwDISvgAALISPgCArIQPACAr4QMAyEr4AACyEj4AgKyEDwAgK+GDLmH48OHpjDPO6OhidHrXXnttev/735+6d++eDjvssFRrXCeQh/BBzbnllltSt27d0rJly9p9f8yYMWnkyJHveTv/+I//mL797W+/5/V0Ff/0T/+ULrroonTcccel+fPnp6uvvnqry8YNPs5heerVq1f63d/93XTZZZelt99+O2u5d2Vtj9Mee+xRhLuTTz45/f3f/33avHnzTq97wYIFafbs2RUtL2yvHtu9JNSwVatWpd12222Hw8ecOXMEkO300EMPFcf4r//6r1PPnj3fdfkIHH/1V39V/H9DQ0O6++670xVXXJF+9atfpdtuuy1DiWtDy+P0P//zP+n5559P//AP/1AEkAjacdz69++/U+HjySefTNOmTatCqWHbhA+6zB/wWrNhw4bUt2/fVCvWrVuX+vTps13BI/To0SN98YtfbH79ta99LR177LHp9ttvT9dff33ab7/9qlja2tH2OIUrr7wyzZo1K9XX16ezzjor3XnnnR1WPtgZml3oEtq25W/cuDHNmDEjHXTQQal3795p4MCB6WMf+1h68MEHi/dj2aj1CC2rvVsGg69//etp6NChRbD5vd/7vXTdddeltl8SHZ9Uzz///LT33nunfv36pU9/+tPphRdeKNbVskYl/j/m/ed//mf6whe+kPbaa6+iPOHf//3fi/JEdXuUdf/9909nnnlm+u///u9W2yqv4+mnny5uVnV1dWmfffZJl156aVGutWvXppNOOqn4lBzr+PM///PtOna//e1vixqJAw88sNjXOJaXXHJJampqal4m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rIQPACAr4QMAyEr4AACyEj4AgKyEDwAgK+EDAMhK+AAAshI+AICshA8AICvhAwDISvgAALISPgCArIQPACAr4QMAyEr4AACyEj4AgKyEDwAgK+EDAMhK+AAAshI+AICshA8AICvhAwDYtcPH4sWL08SJE9PgwYNTt27d0l133dXq/VKplC677LI0aNCg1KdPnzRu3Lj0zDPPVLLMAEBXCh8bNmxIo0aNSnPmzGn3/e9+97vpL/7iL9K8efPSL37xi9S3b980fvz49Pbbb1eivABAjeuxo79w4oknFlN7otZj9uzZ6Vvf+lY66aSTink/+MEP0n777VfUkJx66qnvvcQAQE2raJ+P5557Lr300ktFU0tZXV1dOvroo9OSJUva/Z2mpqbU2NjYagIAOq+Kho8IHiFqOlqK1+X32po5c2YRUMrT0KFDK1kkAGAX0+GjXerr61NDQ0PztHbt2o4uEgBQK+Fj//33L36+/PLLrebH6/J7bfXq1Sv179+/1QQAdF4VDR8jRowoQsaiRYua50Ufjhj1cswxx1RyUwBAVxntsn79+rR69epWnUxXrlyZBgwYkIYNG5amTZuWrrzyynTQQQcVYeTSSy8tngkyadKkSpcdAOgK4WPZsmVp7Nixza8vvPDC4ufkyZPTLbfcki666KLiWSBnn312euONN9LHPvaxdP/996fevXtXtuQAQNcIH2PGjCme57E18dTT73znO8UEALDLjXYBALoW4QMAyEr4AACyEj4AgKyEDwAgK+EDAMhK+AAAshI+AICshA8AICvhAwDISvgAALISPgCArIQPACAr4QMAyEr4AACyEj4AgKyEDwAgK+EDAMhK+AAAshI+AICshA8AICvhAwDIqkfezQGVNHz6fVVb95pZE6q2bqBrU/MBAGQlfAAAWQkfAEBWwgcAkJXwAQBkJXwAAFkJHwBAVsIHAJCV8AEAZCV8AABZCR8AQFbCBwCQlfABAGQlfAAAWQkfAEBWwgcAkJXwAQBkJXwAAFkJHwBAVsIHAJCV8AEAZCV8AABZCR8AQFbCBwCQlfABANR2+Ni0aVO69NJL04gRI1KfPn3SgQcemK644opUKpUqvSkAoAb1qPQKr7nmmjR37tx06623pg996ENp2bJlacqUKamuri6df/75ld4cANDVw8fPf/7zdNJJJ6UJEyYUr4cPH55uv/32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"
    print(
        llm_client.get_single_answer(
            [
                LlmMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Can you describe this graph?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{test_img}"},
                        },
                    ],
                )
            ]
        )
    )
