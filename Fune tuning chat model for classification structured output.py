# Databricks notebook source
# MAGIC %md
# MAGIC ## Config
# MAGIC

# COMMAND ----------

# MAGIC %pip install databricks-genai databricks-sdk openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# TODO: Update this to your database name
catalog = ""
db = ""
cluster_id = ""

registered_model_name = f"{catalog}.{db}.prompt_classification_ft_llama_3_1_8B_chat"
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
base_url = "https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints"

spark.sql(f"USE {catalog}.{db}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a table with sample data

# COMMAND ----------

spark.read.table(f"{catalog}.{db}.tagged_prompts").display()

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql.functions import pandas_udf, to_json
import pandas as pd
import random

system_prompt = """You are a classifier. Write a response that appropriately classifies the user prompt in one of the following classes. Reply with only and exactly the full class name.

Classes:
Sports
News
Music
Politics

"""



@pandas_udf("array<struct<role:string, content:string>>")
def create_conversation(sentence: pd.Series, entities: pd.Series) -> pd.Series:
    def build_message(s, e):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(s)},
            {"role": "assistant", "content": e}]
                
    # Apply build_message to each pair of sentence and entity
    return pd.Series([build_message(s, e) for s, e in zip(sentence, entities)])


df = spark.read.table(f"{catalog}.{db}.tagged_prompts").select(create_conversation("question", "tagged_category").alias("messages"))

display(df)

# COMMAND ----------

df.write.mode("overwrite").saveAsTable("classification_finetuning_chat")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM classification_finetuning_chat

# COMMAND ----------

# MAGIC %md
# MAGIC # Run finetuning now
# MAGIC Fine tune with llama 3.1 8B

# COMMAND ----------

from databricks.model_training import foundation_model as fm

run = fm.create(
  data_prep_cluster_id = cluster_id,  # Required if you are using delta tables as training data source. This is the cluster id that we want to use for our data prep job. See ./_resources for more details
  model=base_model_name,
  train_data_path=f"{catalog}.{db}.classification_finetuning_chat",
  task_type = "CHAT_COMPLETION",
  register_to=registered_model_name,
  training_duration='3ep' # Start with 3 epochs and increase if necessary
)
print(run)

# COMMAND ----------

displayHTML(f'Open the <a href="/ml/experiments/{run.experiment_id}/runs/{run.run_id}/model-metrics">training run on MLflow</a> to track the metrics')
display(run.get_events())

# COMMAND ----------

# MAGIC %md
# MAGIC # Query the serving endpoint 
# MAGIC

# COMMAND ----------

from openai import OpenAI
import os

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": system_prompt
  },
  {
    "role": "user",
    "content": "This is a news about politics"
  }
  ],
  response_format = {
      "type": "json_schema",
      "json_schema": {
        "name": "class",
        "schema": {
          "type": "string",
          "enum": [
            "Sports",
            "News",
            "Music",
            "Politics"
          ],
        },
        "strict": True
      }
    },
  model="databricks-meta-llama-3-1-70b-instruct",
  # TODO: Point this to the served finetuned model
  # model="prompt_classification_ft_llama_3_1_8b_chat",
  max_tokens=256
)


print(chat_completion.choices[0].message.content)

# COMMAND ----------


