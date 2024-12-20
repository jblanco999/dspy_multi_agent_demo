# Databricks notebook source
# MAGIC %pip install --upgrade dspy mlflow databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import dspy
llama = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct', cache=False)
mixtral = dspy.LM('databricks/databricks-mixtral-8x7b-instruct')
gpt4o = dspy.LM('databricks/azure-openai-gpt-4o')
dbrx = dspy.LM('databricks/databricks-dbrx-instruct')
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

from typing import Literal

class Emotion_demo_purposes(dspy.Signature):
    """Example signature showing how you can use differnet typing, add hints or descriptions in the field."""

    sentence: str = dspy.InputField()
    breaker: int = dspy.InputField(description="Add this value to the confidence score")
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()
    confidence: float = dspy.OutputField()

# COMMAND ----------

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

classify = dspy.Predict(Emotion_demo_purposes)
classify(sentence=sentence, breaker=1)

# COMMAND ----------

from pydantic import BaseModel
from typing import List, Callable, Optional, Type
import requests
import time
import mlflow
import pandas as pd

mlflow.dspy.autolog()
# mlflow.create_experiment("Austin_choi_DSPY")
# mlflow.set_experiment("Austin_choi_DSPY")
#Designed a pydantic class to break from the ReAct circle DSPy requires
class Agent(BaseModel):
    signature: Type[dspy.Signature]
    # signature: str
    tools: List[Callable]
    name: Optional[str] = "Agent"
    class Config:
        arbitrary_types_allowed = True

class start_signature(dspy.Signature):
    """Take an input and determine which agent to transfer to. Always finish with an agent transfer call"""

    question: str = dspy.InputField(description="This question contains a pokemon")
    order: str = dspy.OutputField(description="Identify the pokemon in the question and Return the pokemon")

class sales_signature(dspy.Signature):
    """First identify the pokemon, then call the pokemon_lookup tool. Then send the order information with the pokemon information to the execute_order tool.
    Your output should be a summary of the completed order."""

    order: str = dspy.InputField()
    order_confirmation: str = dspy.OutputField(description="This must summarize all the information and confirm order was executed")

class react_multi_agent_example(dspy.Module):
    def __init__(self):
        self.respond = dspy.ReAct(start_signature, tools=[self.transfer_to_sales_agent])
        self.execute = dspy.ReAct(sales_signature, tools=[self.execute_order, self.pokemon_lookup])

    def transfer_to_sales_agent(self):
        sales_agent = Agent(signature=sales_signature, tools=[self.execute_order, self.pokemon_lookup], name="Sales Agent")
        return sales_agent

    def execute_order(self, order: str):
    # if "pikachu" in order.lower():
    #     return "order executed successfully"
    # else:
    #     return "non pokemon order executed successfully"
        return "order executed successfully"

    def pokemon_lookup(self, pokemon_name: str):
        """Use this to lookup a pokemon you don't know. Only send the pokemon name"""
        url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
        response = requests.get(url)
        if response.status_code == 200:
            pokemon_data = response.json()
            pokemon_info = {
                "name": pokemon_data["name"],
                "height": pokemon_data["height"],
                "weight": pokemon_data["weight"],
                "abilities": [ability["ability"]["name"] for ability in pokemon_data["abilities"]],
                "types": [type_data["type"]["name"] for type_data in pokemon_data["types"]],
                "stats_name": [stat['stat']['name'] for stat in pokemon_data["stats"]],
                "stats_no": [stat['base_stat'] for stat in pokemon_data["stats"]]
            }
            results = str(pokemon_info)
            return results
        else:
            return None

    def forward(self, question):
        pokemon_order = self.respond(question=question, tools=[self.transfer_to_sales_agent])
        output = self.execute(order=pokemon_order.order, tools=[self.execute_order, self.pokemon_lookup])
        # return [{'order_confirmation': output.order_confirmation}]
        return output.order_confirmation

response = react_multi_agent_example()
result = response(question="order me our sinistcha")
print(result)


# COMMAND ----------

# signature_input = {"messages": [{"role": "user", "content": "order me a sinistcha"}]}
signature_input = pd.DataFrame({'order': ["order me our sinistcha"]})
# signature_output = pd.DataFrame({'order_confirmation': [result.order_confirmation]})
signature_output = pd.DataFrame({'order_confirmation': [result]})
signature = mlflow.models.infer_signature(model_input=signature_input, model_output=signature_output)
with mlflow.start_run():
    response = react_multi_agent_example()
    model_info = mlflow.dspy.log_model(
        response,
        "dspy_program",
        signature=signature,
        task="llm/v1/chat",
        registered_model_name="austin_choi_demo_catalog.agents.dspy_demo"
    )

# COMMAND ----------

loaded_model = mlflow.dspy.load_model(model_info.model_uri)
# loaded_model = mlflow.dspy.load_model("models:/austin_choi_demo_catalog.agents.dspy_demo/16")
order = 'order me a pancham'
result = loaded_model(order)
# result = loaded_model.forward(order)
print(result)

# COMMAND ----------

import mlflow
import pandas as pd

eval_data = pd.DataFrame({
    "request": [
        "I'd like a pikachu?",
        "I need as torchic",
        "Can you give me a charmander please",
        "I want squirtle!!",
        "Could I have a bulbasaur",
        "gimme an eevee",
        "May I see a jigglypuff",
        "show me snorlax",
        "I would like to see mewtwo",
        "can i have a gengar",
    ],
    "expected_response": [
        "Pikachu order successfully executed. Pikachu is an Electric-type Pokémon with abilities such as Static and Lightning Rod. It has a height of 4 decimetres and a weight of 60 hectograms. Its stats include HP: 35, Attack: 55, Defense: 40, Special Attack: 50, Special Defense: 50, and Speed: 90. The order has been confirmed and completed",
        "Torchic ordered. Torchic is a Fire-type Pokémon with abilities such as Blaze and Speed Boost. It has a height of 4 decimetres and a weight of 25 hectograms. Its stats include HP: 45, Attack: 60, Defense: 40, Special Attack: 70, Special Defense: 50, and Speed: 45. Thank you for your order!",
        "Charmander request fulfilled. Charmander is a Fire-type Pokémon with abilities such as Blaze and Solar Power. It has a height of 6 decimetres and a weight of 85 hectograms. Its stats include HP: 39, Attack: 52, Defense: 43, Special Attack: 60, Special Defense: 50, and Speed: 65. Your request has been processed successfully",
        "Squirtle delivery confirmed. Squirtle is a Water-type Pokémon with abilities such as Torrent and Rain Dish. It has a height of 5 decimetres and a weight of 90 hectograms. Its stats include HP: 44, Attack: 48, Defense: 65, Special Attack: 50, Special Defense: 64, and Speed: 43. Order completed as requested",
        "Bulbasaur has been provided. Bulbasaur is a Grass/Poison-type Pokémon with abilities such as Overgrow and Chlorophyll. It has a height of 7 decimetres and a weight of 69 hectograms. Its stats include HP: 45, Attack: 49, Defense: 49, Special Attack: 65, Special Defense: 65, and Speed: 45. Request fulfilled",
        "Eevee order processed. Eevee is a Normal-type Pokémon with abilities such as Run Away and Adaptability. It has a height of 3 decimetres and a weight of 65 hectograms. Its stats include HP: 55, Attack: 55, Defense: 50, Special Attack: 45, Special Defense: 65, and Speed: 55. Order complete",
        "Jigglypuff request completed. Jigglypuff is a Normal/Fairy-type Pokémon with abilities such as Cute Charm and Competitive. It has a height of 5 decimetres and a weight of 55 hectograms. Its stats include HP: 115, Attack: 45, Defense: 20, Special Attack: 45, Special Defense: 25, and Speed: 20. Request processed",
        "Snorlax display confirmed. Snorlax is a Normal-type Pokémon with abilities such as Immunity and Thick Fat. It has a height of 21 decimetres and a weight of 4600 hectograms. Its stats include HP: 160, Attack: 110, Defense: 65, Special Attack: 65, Special Defense: 110, and Speed: 30. Order fulfilled",
        "Mewtwo showcase complete. Mewtwo is a Psychic-type Pokémon with abilities such as Pressure and Unnerve. It has a height of 20 decimetres and a weight of 1220 hectograms. Its stats include HP: 106, Attack: 110, Defense: 90, Special Attack: 154, Special Defense: 90, and Speed: 130. Request processed",
        "Gengar order executed. Gengar is a Ghost/Poison-type Pokémon with abilities such as Levitate and Cursed Body. It has a height of 15 decimetres and a weight of 405 hectograms. Its stats include HP: 60, Attack: 65, Defense: 60, Special Attack: 130, Special Defense: 75, and Speed: 110. Order complete"
    ]
})

eval_result = mlflow.evaluate(    
    model=loaded_model, 
    data=eval_data,
    model_type="databricks-agent",
)

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_result.tables['eval_results'])

# COMMAND ----------

eval_result.metrics

# COMMAND ----------

print(result)

# COMMAND ----------

print(result.response) #You are essentially calling the variable you defined in the outputfield of the sale_signature

# COMMAND ----------

llama.history

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

trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
react = dspy.ReAct("question -> answer", tools=[search])

tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
optimized_react = tp.compile(react, trainset=trainset)
