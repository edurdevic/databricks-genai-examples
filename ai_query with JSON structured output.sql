-- Databricks notebook source
-- MAGIC %md
-- MAGIC # ai_query with structured output
-- MAGIC
-- MAGIC Example of using JSON `responseFormat` with `ai_query`

-- COMMAND ----------

SELECT
  ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    -- TODO: Update column name
    concat('Extract the email address in JSON format: ', column_name), 
    -- TODO: Update output format
    responseFormat =>
        '{
          "type": "json_schema",
          "json_schema": {
            "name": "extraction",
            "schema": {
              "type": "object",
              "properties": {
                "email": { "type": "string" }
              }
            },
            "strict": true
          }
        }'
  ) AS json
-- TODO: Update table name
FROM mytable 


