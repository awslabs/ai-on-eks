
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



class State(TypedDict):
    messages: Annotated[list, add_messages]
    
# Add new node for external processing
async def external_automation_node(state: State) -> State:
    """
    Node that calls external processing service
    """
    # Get the last message content
    last_message = state["messages"][-1].content
    
    # Call external auto approve service
    
    return {"messages": [last_message]}

async def external_human_node(state: State) -> State:
    """
    Node that calls external processing service
    """
    # Get the last message content
    last_message = state["messages"][-1].content
    
    # Call external auto approve service
    
    return {"messages": [last_message]}