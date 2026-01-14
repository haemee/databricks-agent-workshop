# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Mosaic AI Agent Framework: Author, deploy, and trace a simple agent
# MAGIC
# MAGIC This notebook demonstrates how to build and manage a simple gen AI agent:
# MAGIC - Author a simple gen AI agent with the MLflow 3 `ResponsesAgent` API.
# MAGIC - Manually test the agent, and run batch evaluation using MLflow.
# MAGIC - Log and deploy the agent with Mosaic AI Agent Framework.
# MAGIC - Trace and monitor the agent in real time.
# MAGIC
# MAGIC You can use this pattern with any Agent Framework agent ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/author-agent)).
# MAGIC
# MAGIC MLflow 3 ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai) | [OSS](https://mlflow.org/docs/latest/genai)) provides observability features allowing you to:
# MAGIC - Track quality and operational performance (latency, request volume, errors, etc.)
# MAGIC - Run LLM-based evaluations on production traffic to detect drift or regressions using Agent Evaluation's LLM judges
# MAGIC - Deep dive into individual requests to debug and improve agent responses.
# MAGIC - Transform real-world logs into evaluation sets to drive continuous improvements
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC Address `TODO`s in this notebook before clicking `Run all`.

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Define the agent in code
# MAGIC Define the agent code in a single cell below. This lets you easily write the agent code to a local Python file `agent.py`, using the `%%writefile` magic command, for subsequent logging and deployment.

# COMMAND ----------

# backoff: 재시도 로직 관리
# databricks-openai: OpenAI SDK의 Databricks 확장
# databricks-agents: Mosaic AI 에이전트 프레임워크
# mlflow-skinny[databricks] MLflow의 모든 기능 중에서 핵심적인 트래킹(Tracing)과 로깅(Logging) 기능만 모아둔 가벼운 버전

# COMMAND ----------

# MAGIC %%writefile simple_agent.py
# MAGIC import warnings
# MAGIC from typing import Any, Generator
# MAGIC
# MAGIC import backoff
# MAGIC import mlflow
# MAGIC import openai
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC #LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
# MAGIC #LLM_ENDPOINT_NAME = "databricks-gemini-3-flash"
# MAGIC LLM_ENDPOINT_NAME = "databricks-gpt-5-2"
# MAGIC
# MAGIC # TODO: Update with your system prompt
# MAGIC SYSTEM_PROMPT = """
# MAGIC 너는 질문에 답하는 AI 에이전트야. 항상 한글로 답변해줘. 답변은 명확하고 간단하게 해줘.
# MAGIC """
# MAGIC
# MAGIC
# MAGIC class SimpleChatAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     Simple chat agent that calls an LLM using the Databricks OpenAI client API.
# MAGIC
# MAGIC     You can replace this with your own agent.
# MAGIC     The decorators @mlflow.trace tell MLflow Tracing to track calls to the agent.
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self):
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC         self.client = self.workspace_client.serving_endpoints.get_open_ai_client()
# MAGIC         self.llm_endpoint = LLM_ENDPOINT_NAME
# MAGIC         self.SYSTEM_PROMPT = SYSTEM_PROMPT
# MAGIC
# MAGIC     @backoff.on_exception(backoff.expo, openai.RateLimitError)
# MAGIC     @mlflow.trace(span_type=SpanType.LLM)
# MAGIC     def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
# MAGIC         with warnings.catch_warnings():
# MAGIC             warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
# MAGIC             for chunk in self.client.chat.completions.create(
# MAGIC                 model=self.llm_endpoint,
# MAGIC                 messages=to_chat_completions_input(messages),
# MAGIC                 stream=True,
# MAGIC             ):
# MAGIC                 yield chunk.to_dict()
# MAGIC
# MAGIC     # 동기식 응답을 반환하는 메서드. predict_stream()에서 스트리밍 이벤트를 받고 그 중 reponse.output_item.done 이벤트만 필터링해서 최종 출력 리스트 생성
# MAGIC     # With autologging, you do not need @mlflow.trace here, but you can add it to override the span type.
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     # 스트리밍 방식으로 응답 생성. call_llm()에서 청크 단위로 응답을 받아서 ResponsesAgentStreamEvent로 변환하여 실시간으로 yield
# MAGIC     # With autologging, you do not need @mlflow.trace here, but you can add it to override the span type.
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
# MAGIC             i.model_dump() for i in request.input
# MAGIC         ]
# MAGIC         yield from output_to_responses_items_stream(chunks=self.call_llm(messages))
# MAGIC
# MAGIC # OpenAI SDK를 통한 모든 LLM 호출을 자동으로 추적
# MAGIC mlflow.openai.autolog()
# MAGIC # 인스턴스 생성 
# MAGIC AGENT = SimpleChatAgent()
# MAGIC # 생성된 인스턴스를 MLflow의 현재 모델로 설정, "Models from Code" 방식으로 모델을 로깅 시 어떤 객체를 모델로 사용할지 설정 
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. 
# MAGIC
# MAGIC Since you manually traced methods within `ResponsesAgent`, you can view the trace for each step the agent takes, with any LLM calls made via the OpenAI SDK automatically traced by autologging.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from simple_agent import AGENT

AGENT.predict({"input": [{"role": "user", "content": "What is 5+5?"}]}).model_dump(exclude_none=True)

# COMMAND ----------

# DBTITLE 1,Cell 9
for event in AGENT.predict_stream(
    {"input": [{"role": "user", "content": "What is 5+5?"}]}
):
    print(event.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the agent as an MLflow model, and register it to Unity Catalog
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC
# MAGIC In the same logging call, we can register the model to Unity Catalog, which will be needed for deploying the agent in the next step.  Read the Databricks documentation to learn more about Models in Unity Catalog ([AWS](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/) | [GCP](https://docs.databricks.com/gcp/en/machine-learning/manage-model-lifecycle/)).

# COMMAND ----------

# To-Do: 워크샵용 카탈로그명으로 변경 필요
catalog_name = "hpark_demos"   
user = spark.sql("SELECT current_user()").collect()[0][0]
schema_name = user.split("@")[0].replace("@", "_").replace(".", "_").replace("-", "_")
# schema_name = "ski_agent_workshop"

# 개인 별 스키마 생성
sql = f"""
CREATE SCHEMA IF NOT EXISTS {catalog_name}.`{schema_name}`
"""

# 워크샵에서 사용할 개인 별 카탈로그와 스키마 정보 확인
spark.sql(sql)
print(f"스키마 생성: {catalog_name}.{schema_name}")

# COMMAND ----------

import mlflow
from simple_agent import LLM_ENDPOINT_NAME
from mlflow.models.resources import DatabricksServingEndpoint
from pkg_resources import get_distribution

# The model registry is already set to Databricks Unity Catalog by default,
# but you can change the registry below as needed.
mlflow.set_registry_uri("databricks-uc")

model_name = "simple_agent"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        # Change the model/agent name to be more descriptive for your use case:
        name="simple_agent",
        # Specify the model via the python file created above:
        python_model="simple_agent.py",
        # If you specify pip_requirements instead of extra_pip_requirements,
        # make sure to include mlflow with a version matching this notebook environment.
        extra_pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
        ],
        resources=[DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)],
        # This optional parameter lets you register the model at the same time as logging it:
        registered_model_name=UC_MODEL_NAME,
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment agent validation
# MAGIC Before deploying the agent, perform pre-deployment checks.
# MAGIC
# MAGIC * **Manual vibe checks** using the [mlflow.models.predict() API](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict). See the Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks) | [GCP](https://docs.databricks.com/gcp/en/machine-learning/model-serving/model-serving-debug)).
# MAGIC * **Dataset evaluation checks** using the [mlflow.genai.evaluate() API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate).  See the Databricks documentation ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/evaluate-app) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/evaluate-app) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/evaluate-app)).

# COMMAND ----------

"안녕하세요! 무엇을 도와드릴까요?"

# COMMAND ----------

# Models UI에서 Trace 확인 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch evaluation
# MAGIC
# MAGIC We next demonstrate how to use MLflow to evaluate the agent on a batch of traces.  See Databricks documentation ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/)).
# MAGIC * Collect traces to score.
# MAGIC * Select scorers (LLM judges) to run.
# MAGIC * Run evaluation to compute metrics.

# COMMAND ----------

traces = mlflow.search_traces(max_results=10)

# COMMAND ----------

traces

# COMMAND ----------

from mlflow.genai.scorers import (
    RelevanceToQuery,
    Safety,
    Guidelines,
)

scorers = [
  RelevanceToQuery(),  # 에이전트의 응답이 사용자의 질문(query)과 얼마나 관련성이 있는지 평가
  Safety(),  # 유해 콘텐츠 검사
  # 가이드라인 scorer 는 사용자 지정 가이드라인을 사용하여 응답을 평가
  # Guidelines LLM Judges can evaluate inputs and outputs using pass/fail natural language criteria.
  Guidelines(
      name="concise_communication",
      guidelines="The response MUST be concise and to the point.",
  ),
]

# 위에서 설정한 scorer로 배치 평가 실행
eval_results = mlflow.genai.evaluate(
    data=traces,
    model_id=logged_agent_info.model_id,
    scorers=scorers,
)

# COMMAND ----------

# Guidelines judges보다 더 유연하게 평가하려면, "사용자 지정 프롬프트를 사용하는 LLM Judges"를 활용할 수 있습니다. 
# 다단계 품질 평가(예: excellent/good/poor)와 선택적 점수 부여 등 맞춤형 평가 기준을 제공합니다.

from mlflow.genai.scorers import scorer, ScorerSamplingConfig


@scorer
def formality(inputs, outputs, trace):
    # Must be imported inline within the scorer function body
    from mlflow.genai.judges import custom_prompt_judge
    from mlflow.entities.assessment import DEFAULT_FEEDBACK_NAME

    formality_prompt = """
    You will look at the response and determine the formality of the response.

    <request>{{request}}</request>
    <response>{{response}}</response>

    You must choose one of the following categories.

    [[formal]]: The response is very formal.
    [[semi_formal]]: The response is somewhat formal. The response is somewhat formal if the response mentions friendship, etc.
    [[not_formal]]: The response is not formal.
    """

    my_prompt_judge = custom_prompt_judge(
        name="formality",
        prompt_template=formality_prompt,
        numeric_values={
            "formal": 1,
            "semi_formal": 0.5,
            "not_formal": 0,
        },
        model="databricks:/databricks-gpt-oss-20b",  # optional
    )

    result = my_prompt_judge(request=inputs, response=inputs)
    if hasattr(result, "name"):
        result.name = DEFAULT_FEEDBACK_NAME
    return result

# Register the custom judge and start monitoring
formality_judge = formality.register(name="my_formality_judge")  # name must be unique to experiment
formality_judge = formality_judge.start(sampling_config=ScorerSamplingConfig(sample_rate=0.1))

# COMMAND ----------

# 최대한의 유연성을 위해 LLM 기반 점수를 사용하지 않고도 모니터링을 위한 "사용자 지정 스코어러 함수"를 정의하여 사용할 수 있습니다.

from mlflow.genai.scorers import scorer, ScorerSamplingConfig


# Custom metric: Check if response mentions Databricks
@scorer
def mentions_databricks(outputs):
    """Check if the response mentions Databricks"""
    return "databricks" in str(outputs.get("response", "")).lower()

# Custom metric: Response length check
@scorer(aggregations=["mean", "min", "max"])
def response_length(outputs):
    """Measure response length in characters"""
    return len(str(outputs.get("response", "")))

# Custom metric with multiple inputs
@scorer
def response_relevance_score(inputs, outputs):
    """Score relevance based on keyword matching"""
    query = str(inputs.get("query", "")).lower()
    response = str(outputs.get("response", "")).lower()

    # Simple keyword matching (replace with your logic)
    query_words = set(query.split())
    response_words = set(response.split())

    if not query_words:
        return 0.0

    overlap = len(query_words & response_words)
    return overlap / len(query_words)

# Register and start monitoring custom scorers
databricks_scorer = mentions_databricks.register(name="databricks_mentions")
databricks_scorer = databricks_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))

length_scorer = response_length.register(name="response_length")
length_scorer = length_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

relevance_scorer = response_relevance_score.register(name="response_relevance_score")  # name must be unique to experiment
relevance_scorer = relevance_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

# COMMAND ----------

from mlflow.genai.scorers import Safety, ScorerSamplingConfig

# 스코어러를 이름과 함께 등록하고 모니터링을 시작합니다.
safety_judge = Safety().register(name="my_safety_judge")  # name must be unique to experiment
safety_judge = safety_judge.start(sampling_config=ScorerSamplingConfig(sample_rate=0.7))

# 기본적으로 각 judge는 GenAI 품질 평가를 위해 설계된 Databricks 호스팅 LLM을 사용합니다. scorer 정의에서 model 인자를 사용하여 judge 모델을 Databricks 모델 서빙 엔드포인트로 변경할 수 있습니다. 모델은 반드시 databricks:/<databricks-serving-endpoint-name> 형식으로 지정해야 합니다.
safety_judge = Safety(model="databricks:/databricks-gpt-oss-20b").register(name="my_custom_safety_judge")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent
# MAGIC
# MAGIC Deploy the agent using Agent Framework ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/author-agent)).  This will, by default, log the deployed agent's traces to the current experiment, as well as inference tables (if enabled).

# COMMAND ----------

from databricks import agents

# 배포 시 태그 설정
agents.deploy(UC_MODEL_NAME, model_version=logged_agent_info.registered_model_version, tags={"created": "SKIagentworkshop", "sharable": "true"})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Human Feedback Labeling Session 추가
# MAGIC - developers, end-users 및 domain experts 의 피드백
# MAGIC https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prompt 관리 추가
# MAGIC https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/

# COMMAND ----------

# MAGIC %md
# MAGIC ## View real-time traces from your endpoint
# MAGIC
# MAGIC By default, the deployed agent will log its traces in realtime -to the MLflow Experiment attached to this notebook. To change the MLflow Experiment that contains your traces, call `mlflow.set_experiment(...)` before calling `agents.deploy(...)`.
# MAGIC
# MAGIC You can optionally enable production monitoring to copy traces from the MLflow experiment into a Delta table. ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/production-monitoring)).  If you enable monitoring, then you can visit the MLflow Experiment's **Scorers** tab to update the quality scorers run on your production traces.

# COMMAND ----------

print(f"\nView traces from your endpoint in the MLflow experiment here: https://{mlflow.utils.databricks_utils.get_browser_hostname()}/ml/experiments/{mlflow.get_experiment_by_name(mlflow.utils.databricks_utils.get_notebook_path()).experiment_id}/traces")
