import warnings
from typing import Any, Generator

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

# TODO: Replace with your model serving endpoint
#LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
#LLM_ENDPOINT_NAME = "databricks-gemini-3-flash"
LLM_ENDPOINT_NAME = "databricks-gpt-5-2"

# TODO: Update with your system prompt
SYSTEM_PROMPT = """
너는 질문에 답하는 AI 에이전트야. 항상 한글로 답변해줘. 답변은 명확하고 간단하게 해줘.
"""


class SimpleChatAgent(ResponsesAgent):
    """
    Simple chat agent that calls an LLM using the Databricks OpenAI client API.

    You can replace this with your own agent.
    The decorators @mlflow.trace tell MLflow Tracing to track calls to the agent.
    """

    def __init__(self):
        self.workspace_client = WorkspaceClient()
        self.client = self.workspace_client.serving_endpoints.get_open_ai_client()
        self.llm_endpoint = LLM_ENDPOINT_NAME
        self.SYSTEM_PROMPT = SYSTEM_PROMPT

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                stream=True,
            ):
                yield chunk.to_dict()

    # 동기식 응답을 반환하는 메서드. predict_stream()에서 스트리밍 이벤트를 받고 그 중 reponse.output_item.done 이벤트만 필터링해서 최종 출력 리스트 생성
    # With autologging, you do not need @mlflow.trace here, but you can add it to override the span type.
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    # 스트리밍 방식으로 응답 생성. call_llm()에서 청크 단위로 응답을 받아서 ResponsesAgentStreamEvent로 변환하여 실시간으로 yield
    # With autologging, you do not need @mlflow.trace here, but you can add it to override the span type.
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            i.model_dump() for i in request.input
        ]
        yield from output_to_responses_items_stream(chunks=self.call_llm(messages))

# OpenAI SDK를 통한 모든 LLM 호출을 자동으로 추적
mlflow.openai.autolog()
# 인스턴스 생성 
AGENT = SimpleChatAgent()
# 생성된 인스턴스를 MLflow의 현재 모델로 설정, 로깅 시 "Models from Code" 방식으로 모델을 로깅 시 어떤 객체를 모델로 사용할지 설정 
mlflow.models.set_model(AGENT)
