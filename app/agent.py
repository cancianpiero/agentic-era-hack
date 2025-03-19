# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code="union-attr"
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

GLOBAL_USER_TYPE = "default"


TOOL_PERMISSION = {
    "admin": ["search", 'tool2'],
    "user": ["tool1", 'tool2'], 
}


# 1. Define tools
@tool
def search(query: str) -> str:
    """Simulates a web search. Uses GLOBAL_USER_TYPE to personalize responses."""
    #print(f"GLOBAL_USER_TYPE: {GLOBAL_USER_TYPE}")  # Aggiungiamo un print per debug

    if ("sf" in query.lower() or "san francisco" in query.lower()) and "search" in TOOL_PERMISSION.get(GLOBAL_USER_TYPE, []):
        return f"It's 60 degrees and foggy."
    
    return f"You do not have permission to run this tool."

@tool
def get_product_details(product_name: str):
    """Gathers basic details about a product."""
    details = {
        "smartphone": "A cutting-edge smartphone with advanced camera features and lightning-fast processing.",
        "usb charger": "A super fast and light usb charger",
        "shoes": "High-performance running shoes designed for comfort, support, and speed.",
        "headphones": "Wireless headphones with advanced noise cancellation technology for immersive audio.",
        "speaker": "A voice-controlled smart speaker that plays music, sets alarms, and controls smart home devices.",
    }
    return details.get(product_name, "Product details not found.")


@tool
def get_product_price(product_name: str):
    """Gathers price about a product."""
    details = {
        "smartphone": 500,
        "usb charger": 10,
        "shoes": 100,
        "headphones": 50,
        "speaker": 80,
    }
    return details.get(product_name, "Product price not found.")


from typing import Dict, Any
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState

@tool
def extract_product_info(state: Annotated[dict, InjectedState]) -> Dict[str, Any]:
    """
    Extract product information from a image file or pdf file.
    that was previously uploaded and stored as base64 in the message history.
    Use this tool if there is a multimedial input from the user.
    """
    import base64
    import io
    import os
    import json
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Find the most recent base64 encoded file in message history
    content_to_add = None
    type_list = []
    
    # Iterate through messages in reverse to find the most recent file
    for message in reversed(state["messages"]):
        # Check if message contains file data in base64 format
        for item in message.content:
            if isinstance(item, dict) and "type" in item.keys():
                current_type = item.get("type")
                if current_type in[ "image_url", "media"]:
                    content_to_add = item
                    break
                
    
    if not content_to_add:
        return {"error": "No PDF or image file found in message history", "list_types": state}
    
    try:

        messages = [
            SystemMessage(content="Extract product information including brand, model, specifications, and price. Return the information in JSON format including it inside the xml tag '<json_response>' and '</json_response>."),
            HumanMessage(content=[
                {"type": "text", "text": "Please analyze this product datasheet and extract all relevant information."},
                content_to_add]
                )
            ]

        response = llm.invoke(messages)
        
        # Return the extracted information
        return {"product_info": response.content, "messages": messages}
        
    except Exception as e:
        return {"error": f"Error processing PDF: {str(e)}"}


@tool    
def find_similar_products(dict_info: Dict[str, Any]) -> str:
    """Find similar products given a dictionary of information about a specific product.
    Use this tool after the call of extract_product_info tool to find similar products."""
    import requests
    import json
    import subprocess

    # print(dict_info)
    # Ottieni il token di autenticazione
    token = subprocess.getoutput("gcloud auth print-access-token")

    # URL dell'API
    url = "https://discoveryengine.googleapis.com/v1alpha/projects/458373931597/locations/global/collections/default_collection/engines/rag-products_1742312919541/servingConfigs/default_search:search"

    # Corpo della richiesta
    payload = {
        "query": "can you explain me how to Parallelize the Long-term Memory Training?",
        "pageSize": 10,
        "queryExpansionSpec": {"condition": "AUTO"},
        "spellCorrectionSpec": {"mode": "AUTO"},
        "contentSearchSpec": {
            "extractiveContentSpec": {"maxExtractiveAnswerCount": 1}
        }
    }

    # Intestazioni
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Esegui la richiesta POST
    response = requests.post(url, headers=headers, json=payload)

    # Analizza la risposta JSON
    data = response.json()

    # Estrai solo le risposte alla domanda
    answers = []
    text_answer = ""
    for result in data.get("results", []):
        document_data = result.get("document", {}).get("derivedStructData", {})
        extractive_answers = document_data.get("extractive_answers", [])
        text_answer += "Document name: " + document_data.get("title", "") + "\n"
        for answer in extractive_answers:
            text_answer += f"Page {answer.get('pageNumber')}: {answer.get('content')}" + "\n"
            answers.append(f"Page {answer.get('pageNumber')}: {answer.get('content')}")
        text_answer += "\n"
    
    return text_answer


tools = [extract_product_info, find_similar_products]

# 2. Set up the language model
llm = ChatVertexAI(
    model=LLM, location=LOCATION, temperature=0, max_tokens=1024, streaming=True
).bind_tools(tools)


# 3. Define workflow components
def should_continue(state: MessagesState) -> str:
    """Determines whether to use tools or end the conversation."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """Calls the language model and returns the response."""
    system_message = "You are a helpful AI assistant."
    messages_with_system = [{"type": "system", "content": system_message}] + state[
        "messages"
    ]

    global GLOBAL_USER_TYPE

    ####################################################
    last_message = state["messages"][-1]

    # Verifica se content è una lista di dizionari
    if isinstance(last_message.content, list) and isinstance(last_message.content[0], dict):
        user_type = last_message.content[0].get("user_type", "default")
        GLOBAL_USER_TYPE = user_type
    else:
        user_type = "default"  
        GLOBAL_USER_TYPE = user_type

    ####################################################

    # Forward the RunnableConfig object to ensure the agent is capable of streaming the response.
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}


# 4. Create the workflow graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")

# 5. Define graph edges
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 6. Compile the workflow
agent = workflow.compile()


