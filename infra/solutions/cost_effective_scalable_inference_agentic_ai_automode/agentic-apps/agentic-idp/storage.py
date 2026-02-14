from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

from typing import Annotated, List
from langchain.prompts.chat import HumanMessagePromptTemplate


from typing_extensions import TypedDict

import requests 
import json
import base64

import logging

from langfuse import Langfuse
from datetime import datetime, timedelta
import os
import math
import openai

from PyPDF2 import PdfReader

from pathlib import Path

from langgraph.pregel import RetryPolicy

from decision import  State

# External function to store the data in s3
async def call_store_service(text: str) -> str:
    """
    External function to process text through a service
    """
    try:
        # Example API call - replace with your actual service endpoint
        # response = requests.post(
        #     "http://your-service-endpoint/process",
        #     json={"text": text}
        # )
        # return response.json()
        print(f"Making storage Call with data {text}")
        return {"result": "success"}
    except Exception as e:
        logging.error(f"External service error: {str(e)}")
        return {"error": str(e)}

# Add new node for external processing
async def external_storage_node(state: State) -> State:
    """
    Node that calls external processing service
    """
    # Get the last message content
    # last_message = state["messages"]
    # translated = [state["messages"][0]] + [AIMessage(content=msg.content) for msg in state["messages"][1:]]
    ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    print(f"AI Messages to Store: {json.dumps([msg.content for msg in ai_messages], indent=2)}")

    
    # [-1].content
    print(f"Data to Store {ai_messages}")
    print(json.dumps([msg.content for msg in ai_messages], indent=2))
    
    # Call external service
    result = await call_store_service(json.dumps([msg.content for msg in ai_messages], indent=2))
    
    # Create new message with processed result
    processed_message = HumanMessage(
        content=f"External Processing Results: {json.dumps(result, indent=2)}"
    )
    
    return {"messages": [processed_message]}