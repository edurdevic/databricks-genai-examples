# Databricks notebook source
# MAGIC %md
# MAGIC # NLLB translation
# MAGIC
# MAGIC This notebook downloads the [NLLB model](https://huggingface.co/facebook/nllb-200-distilled-600M) from huggingface, logs it into Unity Catalog and runs a sample prediction.
# MAGIC

# COMMAND ----------

# MAGIC %pip install transformers torch mlflow sentencepiece torchvision

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

model_name = 'facebook/nllb-200-distilled-600M'
registered_model_name = 'erni.models.nllb_200_distilled_600M'

# COMMAND ----------

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model=model_name, src_lang='eng_Latn', tgt_lang="ita_Latn")

# COMMAND ----------

pipe.predict("Hi, how are you doing?")

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.transformers import generate_signature_output
from mlflow.models import infer_signature

input_example = pd.DataFrame(["Hello, I'm a language model,"])
output = generate_signature_output(pipe, input_example)
model_config = { "src_lang": "eng_Latn", "tgt_lang": "ita_Latn"}
signature = infer_signature(input_example, output, params=model_config)


with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=pipe,
        artifact_path="translator_eng_ita",
        model_config=model_config,
        registered_model_name=registered_model_name,
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

# Load the model
my_sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)

my_sentence_generator.predict(
    pd.DataFrame(["Hello, I'm a language model,"]),
    params={"tgt_lang": "fra_Latn"},
)

# COMMAND ----------


