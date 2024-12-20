# Databricks notebook source
# MAGIC %pip install --upgrade dspy mlflow databricks-agents databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set the LLM 
# MAGIC
# MAGIC You can change this to any LLM supported by LiteLLM. Note that this is a global configuration and there are ways to set LLMs for specific modules

# COMMAND ----------

import dspy
llama = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct')
mixtral = dspy.LM('databricks/databricks-mixtral-8x7b-instruct')
gpt4o = dspy.LM('databricks/azure-openai-gpt-4o', cache=False)
dbrx = dspy.LM('databricks/databricks-dbrx-instruct')
llama8b = dspy.LM('databricks/nishant_llama_batch_inference_20241216_1_benchmark', cache=False)
# gpt4o = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_API_KEY')
dspy.configure(lm=llama) #this is to set a default global model

# COMMAND ----------

# MAGIC %md
# MAGIC #DSPy Breakdown
# MAGIC
# MAGIC DSPy is currently made up of a couple core concepts: 
# MAGIC 1. **Signature**: This is declarative structure to define text transformations, specifying what needs to be done rather than how to do it. It enforces Typing in inputs and outputs. Critically, it requires InputField(), OutputField() and Optional Hints/Instructions. DSPy uses the name of the variables as apart of the prompt as well. When it outputs the final JSON, it will use these variable names as apart of the dictionary. I recommend using a class to define the signature with more granularity. Review the Python Typing library to see a full list of types. You can also create your own like a Pydantic BaseModel
# MAGIC 2. **Module**: This is packaged components of DSPy that contain logic for the module's specific task. It is the core of DSPy and enables the ability to modularize your model activities. It requires a forward function to begin the interaction. You can combine multiple modules together to create an entire workflow. Built in modules include COT, Predict and ReAct. You can use the parameter LM to change what model the module uses. 

# COMMAND ----------

import dspy

class rag_signature(dspy.Signature):
    """Retrieve information from the retriever and answer the question"""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    response: str = dspy.OutputField()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using multiple Modules
# MAGIC
# MAGIC You can use different DSPy modules to improve and adjust the performance of your application out of the box. For the example below, I used the base dspy.Predict and dspy.Cot (chain of thought) to see if the reasoning step helps the performance of the application.  
# MAGIC
# MAGIC If you want to use the same index and documents I used here, put the .jsonl files into a vector search index. Run this code to download the file: download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl")

# COMMAND ----------

from dspy import Databricks
from databricks.vector_search.client import VectorSearchClient
import os

class RAG_No_COT(dspy.Module):
    def __init__(self):

        client = VectorSearchClient()

        #set the Databricks vector search index
        self.retriever = client.get_index(
            endpoint_name="one-env-shared-endpoint-12",
            index_name="austin_choi_demo_catalog.rag_demo.ragqa_demo_dspy"
        )

        # Define basic response generator. Predict simply generates an answer
        self.response_generator = dspy.Predict(rag_signature)

    def forward(self, question):

        # Obtain context by executing a Databricks Vector Search query
        retrieved_context = self.retriever.similarity_search(
            query_text=question,
            columns=["doc_id", "text"],
            num_results=3
        )

        # Generate a response using the language model defined in the __init__ method
        response = self.response_generator(context=retrieved_context['result']['data_array'], question=question)
        return response

# COMMAND ----------

from dspy import Databricks
from databricks.vector_search.client import VectorSearchClient
import os

class RAG_COT(dspy.Module):
    def __init__(self):

        client = VectorSearchClient()

        #set the Databricks vector search index
        self.retriever = client.get_index(
            endpoint_name="one-env-shared-endpoint-12",
            index_name="austin_choi_demo_catalog.rag_demo.ragqa_demo_dspy"
        )

        # Define Chain of Thought, an out of the box implementation of having models do reasoning
        self.respond = dspy.ChainOfThought(rag_signature)

    def forward(self, question):

        # Obtain context by executing a Databricks Vector Search query
        retrieved_context = self.retriever.similarity_search(
            query_text=question,
            columns=["doc_id", "text"],
            num_results=3
        )

        # Generate a response using the language model defined in the __init__ method
        response = self.respond(context=retrieved_context['result']['data_array'], question=question)
        return response

# COMMAND ----------

rag_cot = RAG_COT()
rag_no_cot = RAG_No_COT()

result = rag_cot(question="what is the apache spark?")
result

# COMMAND ----------

# MAGIC %md
# MAGIC You can review the query history of your LLM below

# COMMAND ----------

llama.history

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup DSPy Evaluation Datasets
# MAGIC
# MAGIC Using dummy data as an example to illustrate the evaluation technique

# COMMAND ----------

import ujson
from dspy.utils import download

# Download question--answer pairs from the RAG-QA Arena "Tech" dataset.
download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl")

with open("ragqa_arena_tech_examples.jsonl") as f:
    data = [ujson.loads(line) for line in f]

# COMMAND ----------

# MAGIC %md
# MAGIC ###DSPY Example
# MAGIC
# MAGIC These are used to define example inputs that DSPy recognizes and will use to influence evaluations

# COMMAND ----------

data = [dspy.Example(**d).with_inputs('question') for d in data]
example = data[2]
example

# COMMAND ----------

# MAGIC %md
# MAGIC Without COT

# COMMAND ----------

from dspy.evaluate import SemanticF1

# Instantiate the metric.
metric = SemanticF1(decompositional=True)

# Produce a prediction from our `cot` module, using the `example` above as input.
pred = rag_no_cot(**example.inputs())

# Compute the metric score for the prediction.
score = metric(example, pred)

print(f"Question: \t {example.question}\n")
print(f"Gold Response: \t {example.response}\n")
print(f"Predicted Response: \t {pred.response}\n")
print(f"Semantic F1 Score: {score:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC With COT

# COMMAND ----------

from dspy.evaluate import SemanticF1

# Instantiate the metric.
metric = SemanticF1(decompositional=True)

# Produce a prediction from our `cot` module, using the `example` above as input.
pred = rag_cot(**example.inputs())

# Compute the metric score for the prediction.
score = metric(example, pred)

print(f"Question: \t {example.question}\n")
print(f"Gold Response: \t {example.response}\n")
print(f"Predicted Response: \t {pred.response}\n")
print(f"Semantic F1 Score: {score:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create Dataset

# COMMAND ----------

import random

random.Random(0).shuffle(data)
trainset, devset, testset = data[:200], data[200:500], data[500:1000]

len(trainset), len(devset), len(testset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Out of the box Evals
# MAGIC
# MAGIC DSPy has several Out of the box evaluation metrics that you can use. For this, we will just use the example they have in the tutorial called SemanticF1

# COMMAND ----------

# DBTITLE 1,with COT
from dspy.evaluate import SemanticF1
# Define Metric
metric = SemanticF1(decompositional=True)
# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                         display_progress=True, display_table=2)

# Evaluate the Chain-of-Thought program.
evaluate(rag_cot)

# COMMAND ----------

# DBTITLE 1,without COT
from dspy.evaluate import SemanticF1
# Define Metric
metric = SemanticF1(decompositional=True)
# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                         display_progress=True, display_table=2)

# Evaluate the program.
evaluate(rag_no_cot)

# COMMAND ----------


