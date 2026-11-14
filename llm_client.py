from typing import List, Literal, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from string import Template
import asyncio
import httpx
import config

class InputMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(description="Роль")
    content: str = Field(description="Текст сообщения")


class AIAnswer(BaseModel):
    content: str = Field(description="Текст ответа")
    input_tokens: int = Field(description="Количество входных токенов")
    output_tokens: int = Field(description="Количество выходных токенов")

    @property
    def message(self) -> InputMessage:
        return InputMessage(role="assistant", content=self.content)

class OpenAILikeLLM:
    def __init__(self, base_url: str, api_key: str, model: str, common_parameters: dict[str, Any]) -> None:
        custom_http_client = httpx.AsyncClient(trust_env=False)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=custom_http_client)
        self.model = model
        self.common_parameters = common_parameters

    async def generate_answer(self, messages: list[InputMessage], **kwargs: Any) -> AIAnswer:
        response = await self.client.chat.completions.create(
            model=self.model, messages=[msg.model_dump() for msg in messages], **self.common_parameters, **kwargs
        )
        if response.choices[0].message.content is None:
            return AIAnswer(content="Провайдер не отправил текст в ответе", input_tokens=0, output_tokens=0)
        return AIAnswer(
            content=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

# # использование
# async def generate_answer():
#     llm = OpenAILikeLLM(
#         base_url='https://api.groq.com/openai/v1',
#         api_key=config.GROQ_API_KEY,
#         model='openai/gpt-oss-120b',
#         common_parameters={'temperature': 0}
#     )
#
#     query = 'В каком году был придуман инструмент COCI-DROHI?'
#     context = ['COCI-DROHI берут свое начало испокон веков.',
#                'Первое упоминание о COCI-DROHI относится к ледниковому периоду',
#                'COCI-DROHI были придуманы в 7000 году до нашей эры.']
#
#     system_prompt = "Ты полезный ассистент. Используй русский язык для ответа, отвечай строго с использованием контекста, если ответа нет, то скажи, что не можешь ответить, придумывать ничего не надо."
#
#     USER_PROMPT = Template("Вопрос: $query\n\nКонтекст (единственные источники правды):\n$context")
#     user_prompt = USER_PROMPT.substitute({"query": query, "context": "\n\n".join([ek for ek in context])})
#
#     messages = [
#         InputMessage(role="system", content=system_prompt),
#         InputMessage(role="user", content=user_prompt)
#     ]
#     answer = await llm.generate_answer(messages)
#
#     print("Ответ модели:")
#     print(answer.content)
#
#
# if __name__ == "__main__":
#     asyncio.run(generate_answer())