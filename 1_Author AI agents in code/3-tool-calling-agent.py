# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI 에이전트 프레임워크: Mosaic AI Foundation Model API에 호스팅된 모델을 사용하여 OpenAI Responses API 에이전트를 작성하고 배포하기
# MAGIC
# MAGIC 이 노트북에서는 OpenAI Responses 에이전트를 작성하고 [`ResponsesAgent`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent) 인터페이스로 래핑하여 Mosaic AI와 호환되도록 만드는 방법을 보여줍니다. 이 노트북에서 다음을 학습합니다:
# MAGIC
# MAGIC - Mosaic AI Foundation Models를 사용하여 호스팅된 LLM을 호출하는 [Open AI Responses API](https://platform.openai.com/docs/api-reference/responses) 에이전트(ResponsesAgent로 래핑)를 작성합니다.
# MAGIC - 에이전트를 수동으로 테스트합니다
# MAGIC - Mosaic AI 에이전트 평가를 통해 에이전트를 평가합니다
# MAGIC - 에이전트를 기록하고 배포합니다
# MAGIC
# MAGIC Mosaic AI 에이전트 프레임워크를 사용하여 에이전트를 작성하는 방법에 대해 자세히 알아보려면 Databricks 문서([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/create-chat-model))를 참조하세요.
# MAGIC
# MAGIC ## 사전 준비 사항
# MAGIC - 이 노트북의 모든 `TODO`를 해결하세요.

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 코드로 에이전트 정의하기
# MAGIC 아래의 단일 셀에서 에이전트 코드를 정의하세요. 이렇게 하면 `%%writefile` 매직 명령어를 사용하여 에이전트 코드를 로컬 Python 파일로 쉽게 작성할 수 있으며, 이후 로깅 및 배포에 활용할 수 있습니다.
# MAGIC
# MAGIC #### 에이전트 도구
# MAGIC 이 에이전트 코드는 Unity Catalog의 내장 함수인 `system.ai.python_exec`를 에이전트에 추가합니다. 또한, 비정형 데이터 검색을 수행하기 위한 벡터 검색 인덱스를 추가합니다.
# MAGIC
# MAGIC 에이전트에 추가할 수 있는 도구의 더 많은 예시는 Databricks 문서([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/agent-tool))를 참고하세요.

# COMMAND ----------

# MAGIC %%writefile tool_calling_agent.py
# MAGIC import json
# MAGIC import warnings
# MAGIC from typing import Any, Callable, Generator, Optional
# MAGIC from uuid import uuid4
# MAGIC
# MAGIC import backoff
# MAGIC import mlflow
# MAGIC import openai
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC from openai import OpenAI
# MAGIC from pydantic import BaseModel
# MAGIC from unitycatalog.ai.core.base import get_uc_function_client
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
# MAGIC
# MAGIC # TODO: Update with your system prompt
# MAGIC SYSTEM_PROMPT = """
# MAGIC 당신은 Python 코드를 실행할 수 있는 어시스턴트 에이전트입니다.
# MAGIC """
# MAGIC
# MAGIC
# MAGIC # 에이전트가 사용할 도구의 메타데이터를 담는 데이터 클래스
# MAGIC class ToolInfo(BaseModel):
# MAGIC     """
# MAGIC     에이전트의 도구를 나타내는 클래스입니다.
# MAGIC     - "name" (str): 도구의 이름입니다.
# MAGIC     - "spec" (dict): 도구의 JSON 설명(OpenAI Responses 형식과 일치)
# MAGIC     - "exec_fn" (Callable): 도구 로직을 구현하는 함수
# MAGIC     """
# MAGIC
# MAGIC     name: str
# MAGIC     spec: dict
# MAGIC     exec_fn: Callable
# MAGIC
# MAGIC # Unity Catalog UDF를 에이전트 도구로 변환하는 팩토리 함수
# MAGIC def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
# MAGIC     """
# MAGIC     주어진 도구 사양과 (선택적으로) 사용자 정의 실행 함수를 받아 ToolInfo 객체를 생성하는 팩토리 함수입니다.
# MAGIC     """
# MAGIC     # Claude 모델이 지원하지 않는 'strict' 속성을 제거
# MAGIC     tool_spec["function"].pop("strict", None)
# MAGIC     tool_name = tool_spec["function"]["name"]
# MAGIC     # __ 가 포함된 도구 이름을 UDF 점 표기법으로 변환합니다.
# MAGIC     udf_name = tool_name.replace("__", ".")
# MAGIC
# MAGIC     # UC 도구 호출을 위해 kwargs를 받아 UC 도구 실행 클라이언트에 전달하는 래퍼를 정의합니다.
# MAGIC     def exec_fn(**kwargs):
# MAGIC         function_result = uc_function_client.execute_function(udf_name, kwargs)
# MAGIC         # 실행이 실패하면 오류 메시지를, 성공하면 결과 값을 반환합니다.
# MAGIC         if function_result.error is not None:
# MAGIC             return function_result.error
# MAGIC         else:
# MAGIC             return function_result.value
# MAGIC
# MAGIC     # ToolInfo 객체를 반환
# MAGIC     return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)
# MAGIC
# MAGIC
# MAGIC # 에이전트가 사용할 모든 도구 정보를 저장하는 리스트입니다.
# MAGIC TOOL_INFOS = []
# MAGIC
# MAGIC # Unity Catalog의 UDF는 에이전트 도구로 노출될 수 있습니다.
# MAGIC # 아래 코드는 system.ai.python_exec UDF를 사용하여 파이썬 코드 인터프리터 도구를 활성화합니다.
# MAGIC
# MAGIC # TODO: Add additional tools
# MAGIC UC_TOOL_NAMES = ["system.ai.python_exec"]
# MAGIC
# MAGIC uc_function_client = get_uc_function_client()
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC for tool_spec in uc_toolkit.tools:
# MAGIC     TOOL_INFOS.append(create_tool_info(tool_spec))
# MAGIC
# MAGIC
# MAGIC # Databricks 벡터 검색 인덱스를 도구로 사용하기
# MAGIC # 자세한 내용은 https://docs.databricks.com/ko/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge 참고
# MAGIC # 비정형 검색을 위한 벡터 검색 도구 인스턴스를 저장하는 리스트입니다.
# MAGIC VECTOR_SEARCH_TOOLS = []
# MAGIC
# MAGIC # 벡터 검색 검색기 도구를 추가하려면,
# MAGIC # VectorSearchRetrieverTool과 create_tool_info를 사용하여
# MAGIC # 결과를 TOOL_INFOS에 추가하세요.
# MAGIC VECTOR_SEARCH_TOOLS.append(
# MAGIC     VectorSearchRetrieverTool(
# MAGIC         index_name="hpark_demos.ski_agent_workshop.doc_vector_index",
# MAGIC         tool_name="databricks_docs_retriever",
# MAGIC         tool_description="Retrieves customer handling guides from customer handling manual"
# MAGIC         # filters="..."
# MAGIC     )
# MAGIC )
# MAGIC
# MAGIC for vs_tool in VECTOR_SEARCH_TOOLS:
# MAGIC     TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute))
# MAGIC
# MAGIC
# MAGIC class ToolCallingAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     도구 호출 에이전트를 나타내는 클래스입니다.
# MAGIC     exec_fn을 통한 도구 실행과 model serving을 통한 LLM 상호작용을 모두 처리합니다.
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
# MAGIC         """도구들과 함께 ToolCallingAgent를 초기화합니다."""
# MAGIC         self.llm_endpoint = llm_endpoint
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC         self.model_serving_client: OpenAI = (
# MAGIC             self.workspace_client.serving_endpoints.get_open_ai_client()
# MAGIC         )
# MAGIC         self._tools_dict = {tool.name: tool for tool in tools}
# MAGIC
# MAGIC     def get_tool_specs(self) -> list[dict]:
# MAGIC         """OpenAI에서 기대하는 형식으로 도구 사양을 반환합니다."""
# MAGIC         return [tool_info.spec for tool_info in self._tools_dict.values()]
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.TOOL)
# MAGIC     def execute_tool(self, tool_name: str, args: dict) -> Any:
# MAGIC         """주어진 인자를 사용하여 지정된 도구를 실행합니다."""
# MAGIC         return self._tools_dict[tool_name].exec_fn(**args)
# MAGIC
# MAGIC     @backoff.on_exception(backoff.expo, openai.RateLimitError)
# MAGIC     @mlflow.trace(span_type=SpanType.LLM)
# MAGIC     def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
# MAGIC         with warnings.catch_warnings():
# MAGIC             warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
# MAGIC             for chunk in self.model_serving_client.chat.completions.create(
# MAGIC                 model=self.llm_endpoint,
# MAGIC                 messages=to_chat_completions_input(messages),
# MAGIC                 tools=self.get_tool_specs(),
# MAGIC                 stream=True,
# MAGIC             ):
# MAGIC                 yield chunk.to_dict()
# MAGIC
# MAGIC     def handle_tool_call(
# MAGIC         self, tool_call: dict[str, Any], messages: list[dict[str, Any]]
# MAGIC     ) -> ResponsesAgentStreamEvent:
# MAGIC         """
# MAGIC         도구 호출을 실행하고, 실행 결과를 메시지 히스토리에 추가한 뒤, 도구 출력이 포함된 ResponsesStreamEvent를 반환합니다.
# MAGIC         """
# MAGIC         args = json.loads(tool_call["arguments"])
# MAGIC         result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
# MAGIC
# MAGIC         tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
# MAGIC         messages.append(tool_call_output)
# MAGIC         return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)
# MAGIC
# MAGIC     def call_and_run_tools(
# MAGIC         self,
# MAGIC         messages: list[dict[str, Any]],
# MAGIC         max_iter: int = 10,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         for _ in range(max_iter):
# MAGIC             last_msg = messages[-1]
# MAGIC             if last_msg.get("role", None) == "assistant":
# MAGIC                 return
# MAGIC             elif last_msg.get("type", None) == "function_call":
# MAGIC                 yield self.handle_tool_call(last_msg, messages)
# MAGIC             else:
# MAGIC                 yield from output_to_responses_items_stream(
# MAGIC                     chunks=self.call_llm(messages), aggregator=messages
# MAGIC                 )
# MAGIC
# MAGIC         yield ResponsesAgentStreamEvent(
# MAGIC             type="response.output_item.done",
# MAGIC             item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
# MAGIC         )
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         session_id = None
# MAGIC         if request.custom_inputs and "session_id" in request.custom_inputs:
# MAGIC             session_id = request.custom_inputs.get("session_id")
# MAGIC         elif request.context and request.context.conversation_id:
# MAGIC             session_id = request.context.conversation_id
# MAGIC
# MAGIC         if session_id:
# MAGIC             mlflow.update_current_trace(
# MAGIC                 metadata={
# MAGIC                     "mlflow.trace.session": session_id,
# MAGIC                 }
# MAGIC             )
# MAGIC
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         session_id = None
# MAGIC         if request.custom_inputs and "session_id" in request.custom_inputs:
# MAGIC             session_id = request.custom_inputs.get("session_id")
# MAGIC         elif request.context and request.context.conversation_id:
# MAGIC             session_id = request.context.conversation_id
# MAGIC
# MAGIC         if session_id:
# MAGIC             mlflow.update_current_trace(
# MAGIC                 metadata={
# MAGIC                     "mlflow.trace.session": session_id,
# MAGIC                 }
# MAGIC             )
# MAGIC
# MAGIC         messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
# MAGIC             i.model_dump() for i in request.input
# MAGIC         ]
# MAGIC         yield from self.call_and_run_tools(messages=messages)
# MAGIC
# MAGIC
# MAGIC # Log the model using MLflow
# MAGIC mlflow.openai.autolog()
# MAGIC AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 에이전트 테스트
# MAGIC
# MAGIC 에이전트와 상호작용하여 출력을 테스트하세요. `ResponsesAgent` 내에서 메서드를 수동으로 추적했으므로, 에이전트가 수행하는 각 단계의 추적을 볼 수 있습니다. OpenAI SDK를 통해 수행된 모든 LLM 호출은 자동 로깅에 의해 자동으로 추적됩니다.
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from tool_calling_agent import AGENT

result = AGENT.predict({"input": [{"role": "user", "content": "What is 6*7 in Python"}], "custom_inputs": {"session_id": "test-session"}})
print(result.model_dump(exclude_none=True))

# COMMAND ----------


result = AGENT.predict({"input": [{"role": "user", "content": "고객 질문에 답할 담당자가 없을 때는 어떻게 해야돼?"}], "custom_inputs": {"session_id": "test-session"}})
print(result.model_dump(exclude_none=True))

# COMMAND ----------

for chunk in AGENT.predict_stream(
    {"input": [{"role": "user", "content": "What is 6*7 in Python?"}], "custom_inputs": {"session_id": "test-session-stream"}}
):
    print(chunk.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 에이전트를 MLflow 모델로 기록하기
# MAGIC
# MAGIC `tool_calling_agent.py` 파일의 코드에서 에이전트를 기록하세요. 자세한 내용은 [MLflow - 코드에서 모델 기록](https://mlflow.org/docs/latest/models.html#models-from-code)을 참고하세요.
# MAGIC
# MAGIC ### Databricks 리소스에 대한 자동 인증 활성화
# MAGIC 가장 일반적인 Databricks 리소스 유형의 경우, 에이전트 로깅 시 리소스 종속성을 미리 선언하는 것이 Databricks에서 지원 및 권장됩니다. 이를 통해 에이전트를 배포할 때 자동 인증 패스스루가 활성화됩니다. 자동 인증 패스스루를 사용하면 Databricks가 에이전트 엔드포인트 내에서 이러한 리소스 종속성에 안전하게 액세스할 수 있도록 단기 자격 증명을 자동으로 프로비저닝, 회전 및 관리합니다.
# MAGIC
# MAGIC 자동 인증을 활성화하려면 `mlflow.pyfunc.log_model()`을 호출할 때 종속 Databricks 리소스를 지정하세요.
# MAGIC
# MAGIC   - **TODO**: Unity Catalog 도구가 [벡터 검색 인덱스](docs link)를 쿼리하거나 [외부 함수](docs link)를 사용하는 경우, 종속 벡터 검색 인덱스와 UC 연결 객체를 각각 리소스로 포함해야 합니다. 자세한 내용은 문서([AWS](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#resources))를 참고하세요.

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
from tool_calling_agent import UC_TOOL_NAMES, VECTOR_SEARCH_TOOLS
import mlflow
from mlflow.models.resources import DatabricksFunction
from pkg_resources import get_distribution

resources = []
for tool in VECTOR_SEARCH_TOOLS:
    resources.extend(tool.resources)
for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="tool_calling_agent.py",
        pip_requirements=[
            "databricks-openai",
            "backoff",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
        ],
        resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 에이전트 평가: Agent Evaluation 사용
# MAGIC
# MAGIC Mosaic AI Agent Evaluation을 사용하여 에이전트의 응답을 기대 응답 및 기타 평가 기준에 따라 평가하세요. 지정한 평가 기준을 활용해 반복적으로 개선하고, MLflow를 통해 산출된 품질 지표를 추적할 수 있습니다.
# MAGIC 자세한 내용은 Databricks 문서([AWS](https://docs.databricks.com/aws/generative-ai/agent-evaluation) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/))를 참고하세요.
# MAGIC
# MAGIC 툴 호출 평가를 위해 커스텀 지표를 추가할 수 있습니다. 자세한 내용은 Databricks 문서([AWS](https://docs.databricks.com/en/generative-ai/agent-evaluation/custom-metrics.html#evaluating-tool-calls) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/custom-metrics#evaluating-tool-calls))를 참고하세요.

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance, Safety

eval_dataset = [
    {
        "inputs": {"input": [{"role": "user", "content": "Calculate the 15th Fibonacci number"}]},
        "expected_response": "The 15th Fibonacci number is 610.",
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input, "custom_inputs": {"session_id": "evaluation-session"}}),
    scorers=[RelevanceToQuery(), Safety()],  # add more scorers here if they're applicable
)

# Review the evaluation results in the MLfLow UI (see console output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 사전 배포 에이전트 검증
# MAGIC 에이전트를 등록하고 배포하기 전에 [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API를 사용하여 사전 배포 검증을 수행하세요. Databricks 문서([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks))를 참고하세요.

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "Hello!"}], "custom_inputs": {"session_id": "validation-session"}},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog에 모델 등록하기
# MAGIC
# MAGIC 에이전트를 배포하기 전에, 반드시 에이전트를 Unity Catalog에 등록해야 합니다.
# MAGIC
# MAGIC - **TODO** 아래의 `catalog`, `schema`, `model_name`을 업데이트하여 MLflow 모델을 Unity Catalog에 등록하세요.

# COMMAND ----------

# To-Do: 워크샵용 카탈로그명으로 변경 필요
catalog_name = "hpark_demos"   
user = spark.sql("SELECT current_user()").collect()[0][0]
schema_name = user.split("@")[0].replace("@", "_").replace(".", "_").replace("-", "_")
schema_name = "ski_agent_workshop"

# 개인 별 스키마 생성
sql = f"""
CREATE SCHEMA IF NOT EXISTS {catalog_name}.`{schema_name}`
"""

# 워크샵에서 사용할 개인 별 카탈로그와 스키마 정보 확인
spark.sql(sql)
print(f"스키마 생성: {catalog_name}.{schema_name}")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

model_name = "tool_calling_agent"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 에이전트 배포하기

# COMMAND ----------

from databricks import agents

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    tags={"endpointSource": "docs"},
    deploy_feedback_model=False,
)
