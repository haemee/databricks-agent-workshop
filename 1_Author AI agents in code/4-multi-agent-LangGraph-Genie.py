# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI 에이전트 프레임워크: Genie 및 서빙 엔드포인트와 함께 멀티 에이전트 시스템 작성 및 배포
# MAGIC
# MAGIC 이 노트북은 Mosaic AI Agent Framework와 [LangGraph](https://blog.langchain.dev/langgraph-multi-agent-workflows/)를 사용하여 멀티 에이전트 시스템을 구축하는 방법을 보여줍니다. 여기서 [Genie](https://www.databricks.com/product/ai-bi/genie)는 에이전트 중 하나입니다.
# MAGIC 이 노트북에서 수행하는 작업:
# MAGIC 1. LangGraph를 사용하여 멀티 에이전트 시스템을 작성합니다.
# MAGIC 1. LangGraph 에이전트를 MLflow `ResponsesAgent`로 래핑하여 Databricks 기능과 호환되도록 합니다.
# MAGIC 1. 멀티 에이전트 시스템의 출력을 수동으로 테스트합니다.
# MAGIC 1. 멀티 에이전트 시스템을 로깅하고 배포합니다.
# MAGIC
# MAGIC 이 예제는 [LangGraph 문서 - 멀티 에이전트 슈퍼바이저 예제](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.md)를 기반으로 합니다.
# MAGIC
# MAGIC ## Genie 에이전트를 사용하는 이유
# MAGIC
# MAGIC 멀티 에이전트 시스템은 각각 특화된 기능을 가진 여러 AI 에이전트로 구성됩니다. 그 중 하나인 Genie는 사용자가 자연어로 구조화된 데이터와 상호작용할 수 있도록 해줍니다. SQL 함수는 미리 정의된 쿼리만 실행할 수 있지만, Genie는 사용자 질문에 답하기 위해 새로운 쿼리를 생성할 수 있는 유연성을 제공합니다.
# MAGIC
# MAGIC ## 사전 준비 사항
# MAGIC
# MAGIC - 이 노트북의 모든 `TODO`를 해결하세요.
# MAGIC - Genie Space를 생성하세요. Databricks 문서를 참고하세요 ([AWS](https://docs.databricks.com/aws/genie/set-up) | [Azure](https://learn.microsoft.com/azure/databricks/genie/set-up)).

# COMMAND ----------

# MAGIC %pip install -U -qqq langgraph-supervisor==0.0.30 mlflow[databricks] databricks-langchain databricks-agents uv 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 멀티 에이전트 시스템 정의
# MAGIC
# MAGIC LangGraph에서 슈퍼바이저 에이전트 노드와 다음 중 하나 이상의 서브에이전트를 사용하여 멀티 에이전트 시스템을 만듭니다:
# MAGIC - **GenieAgent**: Genie Space와 쉽게 상호작용하여 구조화된 데이터를 쿼리할 수 있는 LangChain 러너블.
# MAGIC - **커스텀 서빙 에이전트**: Databricks에 이미 호스팅된 엔드포인트로 동작하는 에이전트.
# MAGIC - **코드 내 툴 호출 에이전트**: 이 노트북 내에서 정의된 Unity Catalog 함수 툴을 호출하는 에이전트. 이 예제에서는 `system.ai.python_exec`를 사용하지만, 추가 가능한 다른 툴 예시는 Databricks 문서를 참고하세요 ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/agent-tool)).
# MAGIC
# MAGIC 슈퍼바이저 에이전트는 각 서브에이전트에 툴 호출을 생성하고 라우팅하며, 필요한 컨텍스트만 전달하는 역할을 합니다. 이 동작을 수정하여 전체 메시지 히스토리를 전달할 수도 있습니다. 자세한 내용은 [LangGraph 문서](https://langchain-ai.github.io/langgraph/reference/supervisor/)를 참고하세요.
# MAGIC
# MAGIC ### 에이전트 코드를 파일로 작성
# MAGIC
# MAGIC 아래 셀에서 에이전트 코드를 정의하세요. 이렇게 하면 `%%writefile` 매직 커맨드를 사용해 에이전트 코드를 로컬 Python 파일로 작성할 수 있으며, 이후 로깅 및 배포에 활용할 수 있습니다.

# COMMAND ----------

# DBTITLE 1,Untitled
# MAGIC %%writefile multi_agent.py
# MAGIC import json
# MAGIC from typing import Generator, Literal
# MAGIC from uuid import uuid4
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC     set_uc_function_client,
# MAGIC )
# MAGIC from databricks_langchain.genie import GenieAgent
# MAGIC from langchain_core.runnables import Runnable
# MAGIC from langchain.agents import create_agent
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph_supervisor import create_supervisor
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC from pydantic import BaseModel
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC
# MAGIC ########################################
# MAGIC # Create your LangGraph Supervisor Agent
# MAGIC ########################################
# MAGIC
# MAGIC GENIE = "genie"
# MAGIC
# MAGIC
# MAGIC class ServedSubAgent(BaseModel):
# MAGIC     endpoint_name: str
# MAGIC     name: str
# MAGIC     task: Literal["agent/v1/responses", "agent/v1/chat", "agent/v2/chat"]
# MAGIC     description: str
# MAGIC
# MAGIC
# MAGIC class Genie(BaseModel):
# MAGIC     space_id: str
# MAGIC     name: str
# MAGIC     task: str = GENIE
# MAGIC     description: str
# MAGIC
# MAGIC
# MAGIC class InCodeSubAgent(BaseModel):
# MAGIC     tools: list[str]
# MAGIC     name: str
# MAGIC     description: str
# MAGIC
# MAGIC
# MAGIC TOOLS = []
# MAGIC
# MAGIC
# MAGIC def stringify_content(state):
# MAGIC     msgs = state["messages"]
# MAGIC     if isinstance(msgs[-1].content, list):
# MAGIC         msgs[-1].content = json.dumps(msgs[-1].content, indent=4)
# MAGIC     return {"messages": msgs}
# MAGIC
# MAGIC
# MAGIC def create_langgraph_supervisor(
# MAGIC     llm: Runnable,
# MAGIC     externally_served_agents: list[ServedSubAgent] = [],
# MAGIC     in_code_agents: list[InCodeSubAgent] = [],
# MAGIC ):
# MAGIC     agents = []
# MAGIC     agent_descriptions = ""
# MAGIC
# MAGIC     # Process inline code agents
# MAGIC     for agent in in_code_agents:
# MAGIC         agent_descriptions += f"- {agent.name}: {agent.description}\n"
# MAGIC         uc_toolkit = UCFunctionToolkit(function_names=agent.tools)
# MAGIC         TOOLS.extend(uc_toolkit.tools)
# MAGIC         agents.append(create_agent(llm, tools=uc_toolkit.tools, name=agent.name))
# MAGIC
# MAGIC     # Process served endpoints and Genie Spaces
# MAGIC     for agent in externally_served_agents:
# MAGIC         agent_descriptions += f"- {agent.name}: {agent.description}\n"
# MAGIC         if isinstance(agent, Genie):
# MAGIC             # to better control the messages sent to the genie agent, you can use the `message_processor` param: https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_langchain.html#databricks_langchain.GenieAgent
# MAGIC             genie_agent = GenieAgent(
# MAGIC                 genie_space_id=agent.space_id,
# MAGIC                 genie_agent_name=agent.name,
# MAGIC                 description=agent.description,
# MAGIC             )
# MAGIC             genie_agent.name = agent.name
# MAGIC             agents.append(genie_agent)
# MAGIC         else:
# MAGIC             model = ChatDatabricks(
# MAGIC                 endpoint=agent.endpoint_name, use_responses_api="responses" in agent.task
# MAGIC             )
# MAGIC             # Disable streaming for subagents for ease of parsing
# MAGIC             model._stream = lambda x: model._stream(**x, stream=False)
# MAGIC             agents.append(
# MAGIC                 create_agent(
# MAGIC                     model,
# MAGIC                     tools=[],
# MAGIC                     name=agent.name,
# MAGIC                     post_model_hook=stringify_content,
# MAGIC                 )
# MAGIC             )
# MAGIC
# MAGIC     # TODO: The supervisor prompt includes agent names/descriptions as well as general
# MAGIC     # instructions. You can modify this to improve quality or provide custom instructions.
# MAGIC     prompt = f"""
# MAGIC     You are a supervisor in a multi-agent system.
# MAGIC
# MAGIC     1. Understand the user's last request
# MAGIC     2. Read through the entire chat history.
# MAGIC     3. If the answer to the user's last request is present in chat history, answer using information in the history.
# MAGIC     4. If the answer is not in the history, from the below list of agents, determine which agent is best suited to answer the question.
# MAGIC     5. Provide a summarized response to the user's last query, even if it's been answered before.
# MAGIC
# MAGIC     {agent_descriptions}"""
# MAGIC
# MAGIC     return create_supervisor(
# MAGIC         agents=agents,
# MAGIC         model=llm,
# MAGIC         prompt=prompt,
# MAGIC         add_handoff_messages=False,
# MAGIC         output_mode="full_history",
# MAGIC     ).compile()
# MAGIC
# MAGIC
# MAGIC ##########################################
# MAGIC # Wrap LangGraph Supervisor as a ResponsesAgent
# MAGIC ##########################################
# MAGIC
# MAGIC
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC         first_message = True
# MAGIC         seen_ids = set()
# MAGIC
# MAGIC         # can adjust `recursion_limit` to limit looping: https://docs.langchain.com/oss/python/langgraph/GRAPH_RECURSION_LIMIT#troubleshooting
# MAGIC         for _, events in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates"]):
# MAGIC             new_msgs = [
# MAGIC                 msg
# MAGIC                 for v in events.values()
# MAGIC                 for msg in v.get("messages", [])
# MAGIC                 if msg.id not in seen_ids
# MAGIC             ]
# MAGIC             if first_message:
# MAGIC                 seen_ids.update(msg.id for msg in new_msgs[: len(cc_msgs)])
# MAGIC                 new_msgs = new_msgs[len(cc_msgs) :]
# MAGIC                 first_message = False
# MAGIC             else:
# MAGIC                 seen_ids.update(msg.id for msg in new_msgs)
# MAGIC                 node_name = tuple(events.keys())[0]  # assumes one name per node
# MAGIC                 yield ResponsesAgentStreamEvent(
# MAGIC                     type="response.output_item.done",
# MAGIC                     item=self.create_text_output_item(
# MAGIC                         text=f"<name>{node_name}</name>", id=str(uuid4())
# MAGIC                     ),
# MAGIC                 )
# MAGIC             if len(new_msgs) > 0:
# MAGIC                 yield from output_to_responses_items_stream(new_msgs)
# MAGIC
# MAGIC
# MAGIC #######################################################
# MAGIC # Configure the Foundation Model and Serving Sub-Agents
# MAGIC #######################################################
# MAGIC
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC # TODO: Add the necessary information about each of your subagents. Subagents could be agents deployed to Model Serving endpoints or Genie Space subagents.
# MAGIC # Your agent descriptions are crucial for improving quality. Include as much detail as possible.
# MAGIC EXTERNALLY_SERVED_AGENTS = [
# MAGIC     # Genie(
# MAGIC     #     space_id="<your_genie_space_id>",
# MAGIC     #     name="<your-genie-name>",
# MAGIC     #     description="This agent can answer questions...",
# MAGIC     # ),
# MAGIC     # ServedSubAgent(
# MAGIC     #     endpoint_name="cities-agent",
# MAGIC     #     name="city-agent", # choose a semantically relevant name for your agent
# MAGIC     #     task="agent/v1/responses",
# MAGIC     #     description="This agent can answer questions about the best cities to visit in the world.",
# MAGIC     # ),
# MAGIC ]
# MAGIC
# MAGIC ############################################################
# MAGIC # Create additional agents in code
# MAGIC ############################################################
# MAGIC
# MAGIC # TODO: Fill the following with UC function-calling agents. The tools parameter is a list of UC function names that you want your agent to call.
# MAGIC IN_CODE_AGENTS = [
# MAGIC     InCodeSubAgent(
# MAGIC         tools=["system.ai.*"],
# MAGIC         name="code execution agent",
# MAGIC         description="The code execution agent specializes in solving programming challenges, generating code snippets, debugging issues, and explaining complex coding concepts.",
# MAGIC     )
# MAGIC ]
# MAGIC
# MAGIC #################################################
# MAGIC # Create supervisor and set up MLflow for tracing
# MAGIC #################################################
# MAGIC
# MAGIC supervisor = create_langgraph_supervisor(llm, EXTERNALLY_SERVED_AGENTS, IN_CODE_AGENTS)
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC AGENT = LangGraphResponsesAgent(supervisor)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Even if you didn't add any subagents in the agent definition above, the supervisor agent can still answer questions. It just won't have any subagents to switch to.
# MAGIC
# MAGIC **Important:** LangGraph internally uses exceptions (something like `Command` or `ParentCommand`) to switch between nodes. These particular exceptions may appear in your MLflow traces as Events, but this behavior is expected and should not be a cause for concern.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from multi_agent import AGENT

# TODO: Replace this placeholder `input_example` with a domain-specific prompt for your agent.
input_example = {
    "input": [
        {"role": "user", "content": "what tools do you have access to"}
    ]
}


AGENT.predict(input_example)

# COMMAND ----------

for event in AGENT.predict_stream(input_example):
  print(event.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the agent as an MLflow model
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC
# MAGIC ### Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends declaring resource dependencies for the agent upfront during logging. This enables automatic authentication passthrough when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling `mlflow.pyfunc.log_model().`
# MAGIC   - **TODO**: If your Unity Catalog tool queries a [vector search index](docs link) or leverages [external functions](docs link), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See docs ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-authentication#supported-resources-for-automatic-authentication-passthrough) | [Azure](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-authentication#supported-resources-for-automatic-authentication-passthrough)).
# MAGIC
# MAGIC   - **TODO**: Add the SQL Warehouse or tables powering your Genie space to enable passthrough authentication. ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-authentication#supported-resources-for-automatic-authentication-passthrough) | [Azure](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-authentication#supported-resources-for-automatic-authentication-passthrough)). If your genie space uses "embedded credentials" then you do not have to add this.

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from multi_agent import EXTERNALLY_SERVED_AGENTS, LLM_ENDPOINT_NAME, TOOLS, Genie
from databricks_langchain import UnityCatalogTool, VectorSearchRetrieverTool
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable
)
from pkg_resources import get_distribution

# TODO: Manually include underlying resources if needed. See the TODO in the markdown above for more information.
resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
# TODO: Add SQL Warehouses and delta tables powering the Genie Space
resources.append(DatabricksSQLWarehouse(warehouse_id="<your_warehouse_id>"))
resources.append(DatabricksTable(table_name="<your_catalog>.<schema>.<table_name>"))

# Add tools from Unity Catalog
for tool in TOOLS:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

# Add serving endpoints and Genie Spaces
for agent in EXTERNALLY_SERVED_AGENTS:
    if isinstance(agent, Genie):
        resources.append(DatabricksGenieSpace(genie_space_id=agent.space_id))
    else:
        resources.append(DatabricksServingEndpoint(endpoint_name=agent.endpoint_name))

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="multi_agent.py",
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"langgraph-supervisor=={get_distribution('langgraph-supervisor').version}",
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)).

# COMMAND ----------

import mlflow
mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data=input_example,
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

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

model_name = "multi_agent"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags={"endpointSource": "docs"}, deploy_feedback_model=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See Databricks documentation ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)).
