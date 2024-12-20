# Databricks notebook source
# MAGIC %pip install --upgrade dspy mlflow databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #DSPy Breakdown
# MAGIC
# MAGIC DSPy is currently made up of a couple core concepts: 
# MAGIC 1. **Signature**: This is declarative structure to define text transformations, specifying what needs to be done rather than how to do it. It enforces Typing in inputs and outputs. Critically, it requires InputField(), OutputField() and Optional Hints/Instructions. DSPy uses the name of the variables as apart of the prompt as well. When it outputs the final JSON, it will use these variable names as apart of the dictionary. I recommend using a class to define the signature with more granularity. Review the Python Typing library to see a full list of types. You can also create your own like a Pydantic BaseModel
# MAGIC 2. **Module**: This is packaged components of DSPy that contain logic for the module's specific task. It is the core of DSPy and enables the ability to modularize your model activities. It requires a forward function to begin the interaction. You can combine multiple modules together to create an entire workflow. Built in modules include COT, Predict and ReAct. You can use the parameter LM to change what model the module uses. 

# COMMAND ----------

# MAGIC %md
# MAGIC #Building Blocks Below! 
# MAGIC
# MAGIC Use the boilerplate code below to try and build your own DSPy workflow

# COMMAND ----------

import dspy
llama = dspy.LM('databricks/databricks-meta-llama-3-1-70b-instruct', cache=False)
mixtral = dspy.LM('databricks/databricks-mixtral-8x7b-instruct')
gpt4o = dspy.LM('databricks/azure-openai-gpt-4o')
dbrx = dspy.LM('databricks/databricks-dbrx-instruct')
gpt4o = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_API_KEY')
#your_model = dspy.LM('check liteLLM for the full list')
dspy.configure(lm=llama) #this is to set a default global model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Signatures
# MAGIC
# MAGIC Define as many as you require depending on your tasks. Each signature should represent an independent task for the langauge model

# COMMAND ----------

from typing import Literal

class name_your_signature(dspy.Signature):
    """Example Docstring. The Docstring is used in the prompt so add some relevant description!"""
    #below are just examples of what you can use. at a minimum have one InputField and one OutputField 
    #make sure the variable names are something relevant to the task
    sentence: str = dspy.InputField()
    breaker: int = dspy.InputField(description="Add this value to the confidence score")
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()
    confidence: float = dspy.OutputField()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modules
# MAGIC Use either the built in modules or create your own to define your own logic for the program to follow. Create as many as required
# MAGIC
# MAGIC See the full list here: https://dspy.ai/learn/programming/modules/
# MAGIC
# MAGIC

# COMMAND ----------

#To use one out of the box:
# function_calling = dspy.ReAct(your_signature, tools=[your_tools])
# chain_of_thought = dspy.cot(your_signature)

#To make your own, follow the following syntax: 
class your_custom_module(dspy.Module):
  """your custom module""" 
  def __init__(self):
        """If you want to chain together other modules within a module, you can declare them in the init function
        Example 1: self.respond = dspy.ReAct(start_signature, tools=[self.transfer_to_sales_agent])
        Example 2: self.execute = dspy.ReAct(sales_signature, tools=[self.execute_order, self.pokemon_lookup])"""

        #TODO: instantiate other modules here if you wish. Or, anything else the class needs

  def any_extra_function_you_want_to_define(self):
    """Any function you wish to define if you plan on doing function calling or just use functions""" 
    #TODO: Add any function logic here. Add multiple functions if you like 

  def forward(self, expected_input, expected_input_2):
    """Required function. Any inputs you want this module to receive must be provided here. When you call this custom module, it will always go to forward first""" 
    #TODO: Your logic here which includes calling other modules, api calls and so forth. Any code you like can go here. 
     

# COMMAND ----------


