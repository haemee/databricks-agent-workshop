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
    # Converts tool name with double underscores to UDF dot notation.
    udf_name = tool_name.replace("__", ".")

    # UC 도구 호출을 위해 kwargs를 받아 UC 도구 실행 클라이언트에 전달하는 래퍼를 정의합니다.
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        # Return error message if execution fails, result value if not.
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value

    # ToolInfo 객체를 반환
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


# List to store information about all tools available to the agent.
TOOL_INFOS = []

# Unity Catalog의 UDF는 에이전트 도구로 노출될 수 있습니다.
# 아래 코드는 system.ai.python_exec UDF를 사용하여 파이썬 코드 인터프리터 도구를 활성화합니다.

# TODO: Add additional tools
UC_TOOL_NAMES = ["system.ai.python_exec"]

uc_function_client = get_uc_function_client()
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))


# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# List to store vector search tool instances for unstructured retrieval.
VECTOR_SEARCH_TOOLS = []

# To add vector search retriever tools,
# use VectorSearchRetrieverTool and create_tool_info,
# then append the result to TOOL_INFOS.
# Example:
# VECTOR_SEARCH_TOOLS.append(
#     VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# )

for vs_tool in VECTOR_SEARCH_TOOLS:
    TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute))


class ToolCallingAgent(ResponsesAgent):
    """
    Class representing a tool-calling Agent.
    Handles both tool execution via exec_fn and LLM interactions via model serving.
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        """Initializes the ToolCallingAgent with tools."""
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
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
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
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
