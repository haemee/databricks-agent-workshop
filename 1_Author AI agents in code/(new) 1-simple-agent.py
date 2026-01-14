# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Mosaic AI 에이전트 프레임워크: 간단한 에이전트 작성, 배포, 추적
# MAGIC
# MAGIC 이 노트북은 간단한 생성형 AI 에이전트의 구축 및 관리 방법을 보여줍니다:
# MAGIC - MLflow 3 `ResponsesAgent` API로 간단한 생성형 AI 에이전트 작성
# MAGIC - 에이전트 수동 테스트 및 MLflow를 활용한 배치 평가 실행
# MAGIC - Mosaic AI Agent Framework로 에이전트 로깅 및 배포
# MAGIC - 에이전트의 실시간 추적 및 모니터링
# MAGIC
# MAGIC 이 패턴은 모든 Agent Framework 에이전트([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/author-agent))에 사용할 수 있습니다.
# MAGIC
# MAGIC MLflow 3([AWS](https://docs.databricks.com/aws/en/mlflow3/genai) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai) | [OSS](https://mlflow.org/docs/latest/genai))는 다음과 같은 관측 기능을 제공합니다:
# MAGIC - 품질 및 운영 성능(지연 시간, 요청량, 오류 등) 추적
# MAGIC - Agent Evaluation의 LLM 판정자를 활용해 생산 트래픽에 대한 LLM 기반 평가 실행, 드리프트 또는 성능 저하 감지
# MAGIC - 개별 요청을 심층 분석하여 에이전트 응답을 디버깅 및 개선
# MAGIC - 실제 로그를 평가 세트로 변환하여 지속적인 개선 추진
# MAGIC
# MAGIC ## 사전 준비 사항
# MAGIC
# MAGIC `TODO`를 모두 해결한 후 `Run all`을 클릭하세요.

# COMMAND ----------

# databricks-openai: OpenAI SDK의 Databricks 확장
# databricks-agents: Mosaic AI 에이전트 프레임워크
# mlflow-skinny[databricks] MLflow의 모든 기능 중에서 핵심적인 트래킹(Tracing)과 로깅(Logging) 기능만 모아둔 가벼운 버전

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 에이전트를 코드로 정의하기
# MAGIC 아래 셀에서 에이전트 코드를 한 번에 정의하세요. 이렇게 하면 `%%writefile` 매직 명령어를 사용해 에이전트 코드를 로컬 Python 파일 `agent.py`로 쉽게 작성할 수 있으며, 이후 로깅 및 배포에 활용할 수 있습니다.

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
# MAGIC # TODO: 모델 서빙 엔드포인트로 교체하세요
# MAGIC #LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
# MAGIC #LLM_ENDPOINT_NAME = "databricks-gemini-3-flash"
# MAGIC LLM_ENDPOINT_NAME = "databricks-gpt-5-2"
# MAGIC
# MAGIC # TODO: 시스템 프롬프트를 업데이트하세요
# MAGIC SYSTEM_PROMPT = """
# MAGIC 너는 질문에 답하는 AI 에이전트야. 항상 한글로 답변해줘. 답변은 명확하고 간단하게 해줘.
# MAGIC """
# MAGIC
# MAGIC
# MAGIC class SimpleChatAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     Databricks OpenAI 클라이언트 API를 사용하여 LLM을 호출하는 간단한 챗 에이전트입니다.
# MAGIC
# MAGIC     필요에 따라 직접 에이전트를 교체할 수 있습니다.
# MAGIC     @mlflow.trace 데코레이터는 에이전트 호출을 MLflow Tracing으로 추적합니다.
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self):
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC         self.client = self.workspace_client.serving_endpoints.get_open_ai_client()
# MAGIC         self.llm_endpoint = LLM_ENDPOINT_NAME
# MAGIC         self.SYSTEM_PROMPT = SYSTEM_PROMPT
# MAGIC
# MAGIC     # backoff: 재시도 로직 관리
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
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     # 스트리밍 방식으로 응답 생성. call_llm()에서 청크 단위로 응답을 받아서 ResponsesAgentStreamEvent로 변환하여 실시간으로 yield
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
# MAGIC ## 에이전트 테스트
# MAGIC
# MAGIC 에이전트와 상호작용하여 출력 결과를 테스트하세요.
# MAGIC
# MAGIC `ResponsesAgent` 내의 메서드를 수동으로 추적했으므로, 에이전트가 각 단계에서 수행하는 추적을 볼 수 있습니다. OpenAI SDK를 통해 이루어진 모든 LLM 호출은 자동 로깅으로 자동 추적됩니다.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from simple_agent import AGENT

AGENT.predict({"input": [{"role": "user", "content": "What is 5+5?"}]}).model_dump(exclude_none=True)

# COMMAND ----------

# DBTITLE 1,Cell 9
for event in AGENT.predict_stream(
    {"input": [{"role": "user", "content": "What is 15+15?"}]}
):
    print(event.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### UI 확인
# MAGIC 1. Experiments 메뉴에서 노트북명으로 Experiment가 생성되었는지 확인 
# MAGIC 2. 해당 Experiment 클릭
# MAGIC 3. Trace에 가서 위에서 테스트한 요청이 추적되었는지 확인

# COMMAND ----------

# MAGIC %md
# MAGIC ### 에이전트를 MLflow 모델로 로깅하고 Unity Catalog에 등록하기
# MAGIC
# MAGIC `agent.py` 파일의 코드를 MLflow 모델로 로깅하세요. 자세한 내용은 [MLflow - 코드 기반 모델](https://mlflow.org/docs/latest/models.html#models-from-code) 문서를 참고하세요.
# MAGIC
# MAGIC 동일한 로깅 호출에서 모델을 Unity Catalog에 등록할 수 있습니다. 이는 다음 단계에서 에이전트를 배포하는 데 필요합니다. Unity Catalog의 모델에 대해 더 알아보려면 Databricks 문서를 참고하세요 ([AWS](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/) | [GCP](https://docs.databricks.com/gcp/en/machine-learning/manage-model-lifecycle/)).

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

# 새로운 Experiment 생성 

import mlflow
from mlflow import MlflowException

experiment_name = f"/Users/{user}/simple_agent_experiment"
try:
    # 2. 실험 생성 시도
    exp_id = mlflow.create_experiment(name=experiment_name)
    print(f"새로운 실험이 생성되었습니다. ID: {exp_id}")
    print(f"경로: {experiment_name}")
    mlflow.set_experiment(experiment_name)
except MlflowException as e:
    # 3. 이미 존재하는 경우 기존 정보 가져오기
    if "already exists" in str(e):
        exp = mlflow.get_experiment_by_name(experiment_name)
        exp_id = exp.experiment_id
        print(f"이미 존재하는 실험입니다. ID: {exp_id}")
        mlflow.set_experiment(experiment_name)
    else:
        raise e


# COMMAND ----------

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
# MAGIC ### UI 확인
# MAGIC 1. Experiments 메뉴에서 "simple_agent_experiment" Experiment가 생성되었는지 확인 
# MAGIC 2. Catalog 메뉴에서  
# MAGIC       카탈로그: 워크샵 카탈로그명 > 스키마: 계정명  아래에 "simple-agent" 모델이 등록되었는지 확인

# COMMAND ----------

# MAGIC %md
# MAGIC ## 배포 전 에이전트 검증
# MAGIC 에이전트 배포 전에 사전 검증을 수행하세요.
# MAGIC
# MAGIC * [mlflow.models.predict() API](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict)를 활용한 **수동 바이브 체크**. Databricks 문서 참고 ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks) | [GCP](https://docs.databricks.com/gcp/en/machine-learning/model-serving/model-serving-debug)).
# MAGIC * [mlflow.genai.evaluate() API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate)를 활용한 **데이터셋 평가 체크**. Databricks 문서 참고 ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/evaluate-app) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/evaluate-app) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/evaluate-app)).

# COMMAND ----------

print(logged_agent_info.registered_model_version)

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/simple_agent",
    input_data={"input": [{"role": "user", "content": "What is 3*3?"}]},
    env_manager="uv",
)

# COMMAND ----------

# DBTITLE 1,Cell 15
mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/simple_agent",
    input_data={"input": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### UI 확인
# MAGIC * Experiments 메뉴에서 "simple_agent_experiment" Experiment에 들어가서 위에서 실행한 두개의 쿼리가 추적되었는지 확인

# COMMAND ----------

# MAGIC %md
# MAGIC ### 배치 평가
# MAGIC
# MAGIC 다음으로 MLflow를 사용하여 에이전트를 배치 트레이스에 대해 평가하는 방법을 보여줍니다. 자세한 내용은 Databricks 문서를 참고하세요 ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/)).
# MAGIC * 평가할 트레이스를 수집합니다.
# MAGIC * 실행할 평가자(LLM 판정자)를 선택합니다.
# MAGIC * 평가를 실행하여 메트릭을 계산합니다.

# COMMAND ----------

# DBTITLE 1,Cell 18
traces = mlflow.search_traces(max_results=10)

# COMMAND ----------

traces

# COMMAND ----------

# MAGIC %md
# MAGIC ## 평가를 위한 Scorer 정의

# COMMAND ----------

from mlflow.genai.scorers import (
    RelevanceToQuery,
    Safety,
    Guidelines,
)

scorers = [
  RelevanceToQuery(),  # 에이전트의 응답이 사용자의 질문(query)과 얼마나 관련성이 있는지 평가
  Safety(),  # 유해 콘텐츠 검사
  # 사용자 지정 가이드라인을 사용하여 Scorer 생성
  # Guidelines LLM Judges는 입력과 출력을 자연어 기준(pass/fail)으로 평가할 수 있습니다.
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

# MAGIC %md
# MAGIC ### UI 확인
# MAGIC * Experiments 메뉴에서 "simple_agent_experiment" Experiment > Trace 에서 각 추적의 평가 결과 확인
# MAGIC

# COMMAND ----------

from mlflow.genai.scorers import Safety, ScorerSamplingConfig

# 스코어러를 이름과 함께 등록하고 모니터링을 시작합니다.
safety_judge = Safety().register(name="my_safety_judge")  # name은 실험마다 고유해야 합니다. 
safety_judge = safety_judge.start(sampling_config=ScorerSamplingConfig(sample_rate=0.7))

# 기본적으로 각 judge는 GenAI 품질 평가를 위해 설계된 Databricks 호스팅 LLM을 사용합니다. scorer 정의에서 model 인자를 사용하여 judge 모델을 Databricks 모델 서빙 엔드포인트로 변경할 수 있습니다. 모델은 반드시 databricks:/<databricks-serving-endpoint-name> 형식으로 지정해야 합니다.
safety_judge = Safety(model="databricks:/databricks-gpt-oss-20b").register(name="my_custom_safety_judge")

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

# MAGIC %md
# MAGIC ### UI 확인
# MAGIC * Experiments 메뉴에서 "simple_agent_experiment" Experiment > Scorer 에서 위에서 생성한 Scorer 가 생성되었는지 확인
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 에이전트 배포
# MAGIC
# MAGIC Agent Framework을 사용하여 에이전트를 배포하세요 ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/author-agent)). 기본적으로, 배포된 에이전트의 트레이스는 현재 실험과 추론 테이블(활성화된 경우)에 기록됩니다.

# COMMAND ----------

from databricks import agents

# 배포 시 태그 설정
agents.deploy(UC_MODEL_NAME, model_version=logged_agent_info.registered_model_version, tags={"created": "SKIagentworkshop", "sharable": "true"})

# COMMAND ----------

# MAGIC %md
# MAGIC ### UI 확인
# MAGIC * Serving에서 엔드포인트가 생성되었는지 확인

# COMMAND ----------

# MAGIC %md
# MAGIC ## 모델 엔드포인트로 질의 하기

# COMMAND ----------

import mlflow.deployments

# 1. Deployment Client 초기화 (Databricks 환경에서는 자동으로 설정됨)
client = mlflow.deployments.get_deploy_client("databricks")

# 2. 엔드포인트 이름 설정
# To-Do: 생성하신 엔드포인트 이름으로 변경하세요.
endpoint_name = "agents_hpark_demos-ski_agent_workshop-simple_agent" 

# 3. 질문(Payload) 구성
# To-Do: 질문을 수정하세요.
input_data = {
    "input": [
        {
            "role": "user", 
            "content": "Databricks AI 에 대해서 설명해줘."
        }
    ],
    "max_output_tokens": 500 
}

# 4. 질문 (Predict/Query)
try:
    response = client.predict(
        endpoint=endpoint_name,
        inputs=input_data
    )

    print("에이전트 응답:")
    print(response)
    
except Exception as e:
    print(f"에러 발생: {e}")
    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Human Feedback Labeling Session 추가
# MAGIC - domain experts 의 피드백
# MAGIC - Expectation > Labeling Schemas 생성 
# MAGIC - Expectation > Labeling Sessions > Create session
# MAGIC
# MAGIC - 참고자료:
# MAGIC https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/

# COMMAND ----------

# MAGIC %md
# MAGIC ## 엔드포인트에서 실시간 트레이스 보기
# MAGIC
# MAGIC 기본적으로, 배포된 에이전트는 이 노트북에 연결된 MLflow Experiment에 실시간으로 트레이스를 기록합니다. 트레이스를 저장할 MLflow Experiment를 변경하려면 `agents.deploy(...)`를 호출하기 전에 `mlflow.set_experiment(...)`를 사용하세요.
# MAGIC
# MAGIC 프로덕션 모니터링을 활성화하면 MLflow Experiment의 트레이스를 Delta 테이블로 복사할 수 있습니다. ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/production-monitoring)). 모니터링을 활성화하면 MLflow Experiment의 **Scorers** 탭에서 프로덕션 트레이스에 대해 실행할 품질 평가자를 업데이트할 수 있습니다.

# COMMAND ----------

print(f"\nView traces from your endpoint in the MLflow experiment here: https://{mlflow.utils.databricks_utils.get_browser_hostname()}/ml/experiments/{mlflow.get_experiment_by_name(mlflow.utils.databricks_utils.get_notebook_path()).experiment_id}/traces")
