import json
import warnings
from typing import Any, Callable, Generator, Optional
from uuid import uuid4

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

############################################
# Define your LLM endpoint and system prompt
############################################
# TODO: Replace with your model serving endpoint
LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"

# TODO: Update with your system prompt
SYSTEM_PROMPT = """
당신은 Python 코드를 실행할 수 있는 어시스턴트 에이전트입니다.
"""


# 에이전트가 사용할 도구의 메타데이터를 담는 데이터 클래스
class ToolInfo(BaseModel):
    """
    에이전트의 도구를 나타내는 클래스입니다.
    - "name" (str): 도구의 이름입니다.
    - "spec" (dict): 도구의 JSON 설명(OpenAI Responses 형식과 일치)
    - "exec_fn" (Callable): 도구 로직을 구현하는 함수
    """

    name: str
    spec: dict
    exec_fn: Callable

# Unity Catalog UDF를 에이전트 도구로 변환하는 팩토리 함수
def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    """
    주어진 도구 사양과 (선택적으로) 사용자 정의 실행 함수를 받아 ToolInfo 객체를 생성하는 팩토리 함수입니다.
    """
    # Claude 모델이 지원하지 않는 'strict' 속성을 제거
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    # __ 가 포함된 도구 이름을 UDF 점 표기법으로 변환합니다.
    udf_name = tool_name.replace("__", ".")

    # UC 도구 호출을 위해 kwargs를 받아 UC 도구 실행 클라이언트에 전달하는 래퍼를 정의합니다.
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        # 실행이 실패하면 오류 메시지를, 성공하면 결과 값을 반환합니다.
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value

    # ToolInfo 객체를 반환
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


# 에이전트가 사용할 모든 도구 정보를 저장하는 리스트입니다.
TOOL_INFOS = []

# Unity Catalog의 UDF는 에이전트 도구로 노출될 수 있습니다.
# 아래 코드는 system.ai.python_exec UDF를 사용하여 파이썬 코드 인터프리터 도구를 활성화합니다.

# TODO: Add additional tools
UC_TOOL_NAMES = ["system.ai.python_exec"]

uc_function_client = get_uc_function_client()
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))


# Databricks 벡터 검색 인덱스를 도구로 사용하기
# 자세한 내용은 https://docs.databricks.com/ko/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge 참고
# 비정형 검색을 위한 벡터 검색 도구 인스턴스를 저장하는 리스트입니다.
VECTOR_SEARCH_TOOLS = []

# 벡터 검색 검색기 도구를 추가하려면,
# VectorSearchRetrieverTool과 create_tool_info를 사용하여
# 결과를 TOOL_INFOS에 추가하세요.
VECTOR_SEARCH_TOOLS.append(
    VectorSearchRetrieverTool(
        index_name="hpark_demos.ski_agent_workshop.doc_vector_index",
        tool_name="databricks_docs_retriever",
        tool_description="Retrieves customer handling guides from customer handling manual"
        # filters="..."
    )
)

for vs_tool in VECTOR_SEARCH_TOOLS:
    TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute))


class ToolCallingAgent(ResponsesAgent):
    """
    도구 호출 에이전트를 나타내는 클래스입니다.
    exec_fn을 통한 도구 실행과 model serving을 통한 LLM 상호작용을 모두 처리합니다.
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        """도구들과 함께 ToolCallingAgent를 초기화합니다."""
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """OpenAI에서 기대하는 형식으로 도구 사양을 반환합니다."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """주어진 인자를 사용하여 지정된 도구를 실행합니다."""
        return self._tools_dict[tool_name].exec_fn(**args)

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            ):
                yield chunk.to_dict()

    def handle_tool_call(
        self, tool_call: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ResponsesAgentStreamEvent:
        """
        도구 호출을 실행하고, 실행 결과를 메시지 히스토리에 추가한 뒤, 도구 출력이 포함된 ResponsesStreamEvent를 반환합니다.
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role", None) == "assistant":
                return
            elif last_msg.get("type", None) == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            i.model_dump() for i in request.input
        ]
        yield from self.call_and_run_tools(messages=messages)


# Log the model using MLflow
mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
mlflow.models.set_model(AGENT)
