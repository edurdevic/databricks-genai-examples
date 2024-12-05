# Databricks notebook source
# MAGIC %md
# MAGIC # Driver notebook
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export. We generated three notebooks in the same folder:
# MAGIC - [agent]($./agent): contains the code to build the agent.
# MAGIC - [config.yml]($./config.yml): contains the configurations.
# MAGIC - [**driver**]($./driver): logs, evaluate, registers, and deploys the agent.
# MAGIC
# MAGIC This notebook uses Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) to deploy the agent defined in the [agent]($./agent) notebook. The notebook does the following:
# MAGIC 1. Logs the agent to MLflow
# MAGIC 2. Registers the agent to Unity Catalog
# MAGIC 3. Deploys the agent to a Model Serving endpoint
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.
# MAGIC - Review the contents of [config.yml]($./config.yml) as it defines the tools available to your agent and the LLM endpoint.
# MAGIC - Review and run the [agent]($./agent) notebook in this folder to view the agent's code, iterate on the code, and test outputs.
# MAGIC
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Log the agent as code from the [agent]($./agent) notebook. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# ---- ADDED ----
 
from dataclasses import dataclass
from typing import Optional, Dict
from mlflow.models.rag_signatures import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
)
from dataclasses import asdict, is_dataclass


@dataclass
class CustomInputs():
    user_email: str = ""

@dataclass
class CustomChatCompletionRequest(ChatCompletionRequest):
    custom_inputs: Optional[CustomInputs] = CustomInputs()
    
# ---- ADDED ----

# COMMAND ----------

# Log the model to MLflow
import os
import mlflow

from mlflow.models import ModelConfig
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
config = ModelConfig(development_config="config.yml")
resources = [DatabricksServingEndpoint(endpoint_name=config.get("llm_endpoint"))]
uc_functions_to_expand = config.get("tools").get("uc_functions")
for func in uc_functions_to_expand:
    # if the function name is a regex, get all functions in the schema
    if func.endswith("*"):
        catalog, schema, _ = func.split(".")
        expanded_functions = list(
            w.functions.list(catalog_name=catalog, schema_name=schema)
        )
        for expanded_function in expanded_functions:
            resources.append(
                DatabricksFunction(function_name=expanded_function.full_name)
            )
    # otherwise just add the function
    else:
        resources.append(DatabricksFunction(function_name=func))

signature = ModelSignature(CustomChatCompletionRequest(), ChatCompletionResponse())

input_example = {
    "messages": [{"role": "user", "content": "What is the solar energy production in EU?"}],
    "custom_inputs": {
        "user_email": "test@example.com"
    }
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        "agent",
        python_model=os.path.join(
            os.getcwd(),
            "agent",
        ),
        signature=signature,
        input_example=input_example,
        model_config="config.yml",
        resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.

# COMMAND ----------

# import pandas as pd

# eval_examples = [
#     {
#         "request": {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": "how much is the solar production capacity in EU?"
#                 }
#             ]
#         },
#         "expected_response": "56 GW"
#     }
# ]

# eval_dataset = pd.DataFrame(eval_examples)
# display(eval_dataset)

# COMMAND ----------

eval_dataset = spark.read.table("erni.daiwt_milano_2024.synthetic_evaluation").withColumnRenamed("inputs", "request").toPandas()
display(eval_dataset)

# COMMAND ----------

import mlflow
import pandas as pd

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",  # replace `chain` with artifact_path that you used when calling log_model.
        data=eval_dataset,  # Your evaluation dataset
        targets="expected_response",
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "erni"
schema = "daiwt_milano_2024"
model_name = "agent_exported_from_playground"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

# Deploy the model to the review app and a model serving endpoint
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version)

# COMMAND ----------


