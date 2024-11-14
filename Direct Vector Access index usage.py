# Databricks notebook source
# MAGIC %md
# MAGIC # Using vector search direct index
# MAGIC
# MAGIC This notebook helps to understand how direct access mode works for Datab ricks vector indexes.
# MAGIC
# MAGIC Direct mode enables to upsert data directly into the vector index without going through a delta index sync.
# MAGIC
# MAGIC In general, the sync approach is the recommended approach, while direct access is an advanced pattern to have more control on the index.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch openai
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# TODO: Update this variables to match your environment
index_name="erni.soc.emails_direct_index"
endpoint_name="one-env-shared-endpoint-12"
embeddings_endpoint="databricks-gte-large-en"
workspace_models_base_url="https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample data

# COMMAND ----------

from pyspark.sql import Row

data = [
    Row(id=i, 
        from_email=f"from_email_{i}@example.com", 
        to_email=f"to_email_{i}@example.com", 
        title=f"Title {i}", 
        content=f"Content of the email {i} which is longer and more descriptive.") 
    for i in range(1, 11)
]

df = spark.createDataFrame(data)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create vector index

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

# COMMAND ----------

# Uncomment to delete if already exists
client.delete_index(index_name=index_name)

# COMMAND ----------

index = client.create_direct_access_index(
  endpoint_name=endpoint_name,
  index_name=index_name,
  primary_key="id",
  embedding_dimension=1024,
  embedding_vector_column="text_vector",
  schema={
    "id": "int",
    "from_email": "string",
    "to_email": "string",
    "title": "string",
    "content": "string",
    "text_vector": "array<float>"}
)

# index = client.get_index(index_name=index_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings

# COMMAND ----------

from pyspark.sql.functions import col, expr

processed_df = df.withColumn("text_vector", expr(f"ai_query('{embeddings_endpoint}', content)"))

processed_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert to index

# COMMAND ----------

pdf = processed_df.toPandas()
pdf["text_vector"] = pdf["text_vector"].apply(lambda x: x.tolist())
index.upsert(pdf.to_dict(orient="records"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query index

# COMMAND ----------

from openai import OpenAI
import os

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=workspace_models_base_url
)

embeddings = client.embeddings.create(
  input='my search query goes here',
  model=embeddings_endpoint
).data[0].embedding


# COMMAND ----------

index.similarity_search(
  query_vector=embeddings,
  num_results=2, 
  columns=['id', 'content']
  )

# COMMAND ----------


