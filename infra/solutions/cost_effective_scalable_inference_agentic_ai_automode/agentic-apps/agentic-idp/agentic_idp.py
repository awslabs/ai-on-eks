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

from doc_reader import encode_image

from exteral_service import external_service_node 
from storage import external_storage_node
from decision import external_automation_node, external_human_node, State

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()



local_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
local_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_host = os.getenv("LANGFUSE_HOST")
if local_public_key is None or local_secret_key is None or langfuse_host is None:
    raise ValueError("Please set the LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST environment variables.")

langfuse = Langfuse(
    secret_key=local_secret_key,
    public_key=local_public_key,
    host=langfuse_host
)


generation_model_key_var = os.getenv("LLAMA_VISION_MODEL_KEY")
api_gateway_url = os.getenv("API_GATEWAY_URL")
if generation_model_key_var is None or api_gateway_url is None:
    raise ValueError("Please set the LLAMA_VISION_MODEL_KEY and API_GATEWAY_URL environment variables.")
generation_model_url = api_gateway_url
generation_model_key = generation_model_key_var

reflection_model_key_var = os.getenv("LLAMA_VISION_MODEL_KEY")
if reflection_model_key_var is None:
    raise ValueError("Please set the LLAMA_VISION_MODEL_KEY environment variable.")
reflection_model_url = api_gateway_url
reflection_model_key = reflection_model_key_var

desired_iterations = 2

reflection_model = "Qwen/QwQ-32B-AWQ"
generation_model = "vllm-server-qwen-vision"





    

# Prompt for ocr Generation
generation_llm = ChatOpenAI(model=generation_model, temperature=0.7, max_tokens=1500, api_key=generation_model_key, base_url=generation_model_url)

client = openai.OpenAI(
    api_key=generation_model_key,             # pass litellm proxy key, if you're using virtual keys
    base_url=api_gateway_url # litellm-proxy-base url
)



# Path to your image
image_path = "birth_cert.png"
base64_image = encode_image(image_path)



generation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an expert birth certificate document processor. Extract and structure information from birth certificates, focusing on key verification fields including name, date of birth, and place of birth. Ensure accurate extraction of hospital or medical facility names for subsequent verification."),
    MessagesPlaceholder(variable_name="messages"),
    ])


# Bind the prompt to the LLM
generate_report = generation_prompt | generation_llm

async def generation_node(state: State) -> State:

    # combined_content = f"{state['messages'][0].content}\n Birth Certificate Image Content:\n {base64_image}"
    # state["messages"][0].content = combined_content    
    response = client.chat.completions.create(
        model="vllm-server-qwen-vision",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is my birth certificate. Extract all the fields from this image and provide the information in a structured json only format, no other text or wrapper around json. The json will be read by machine. The fields include name, date of birth, place of birth. Make sure the output only contains JSON and nothing else. Be strict about it."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )    
    
    # Extract the JSON from the vision model response
    vision_json = response.choices[0].message.content.strip()
    
    # Create a structured response that includes both the original request and the extracted JSON
    structured_response = f"""Birth Certificate Analysis Request: {state['messages'][0].content}

Extracted Birth Certificate Data (JSON):
{vision_json}

Analysis: Based on the extracted birth certificate information, I need to verify the authenticity of this document by validating the place of birth details. The extracted data shows the place of birth as specified in the JSON above, which will be verified against official hospital records and databases."""
    
    return {"messages": [AIMessage(content=structured_response)]}





reflection_llm = ChatOpenAI(model=reflection_model, temperature=0, max_tokens=1000, api_key=reflection_model_key, base_url=reflection_model_url)

# Prompt for Reflection
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert birth certificate verification assessor. Your task is to evaluate birth certificate legitimacy based on place of birth verification results.\n\n"
            "ASSESSMENT CRITERIA:\n"
            "1. PRIMARY FACTOR - Hospital/Place Verification:\n"
            "   - If place_verified=true and confidence_score >= 0.90: High confidence (0.85-0.95)\n"
            "   - If place_verified=true and confidence_score 0.80-0.89: Good confidence (0.75-0.84)\n"
            "   - If place_verified=true and confidence_score 0.70-0.79: Moderate confidence (0.65-0.74)\n"
            "   - If place_verified=false or confidence_score < 0.70: Low confidence (0.20-0.40)\n\n"
            "2. SUPPORTING FACTORS (adjust +/- 0.05):\n"
            "   - Hospital status (Active vs Inactive)\n"
            "   - Verification sources quality\n"
            "   - Contact information availability\n\n"
            "CRITICAL: You must respond with ONLY a valid JSON object in this exact format:\n"
            '{{\"confidence_score\": 0.XX, \"message\": \"explanation here\"}}\n\n'
            "Do not include any other text, thinking, or formatting. Just the JSON object."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Bind the prompt to the LLM
reflect_on_report = reflection_prompt | reflection_llm

# Reflection Node
async def reflection_node(state: State) -> State:
    # Swap message roles for reflection
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    print("***********************")
    print(state["messages"])
    
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    
    # Try multiple times to get proper JSON response
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Create proper input for the prompt template
            prompt_input = {"messages": translated}
            res = await reflect_on_report.ainvoke(prompt_input)
            response_content = res.content.strip()
            
            print(f"Reflection attempt {attempt + 1}: {response_content[:200]}...")
            
            # Check if response contains valid JSON structure
            import re
            if '"confidence_score"' in response_content and '"message"' in response_content:
                # Try to extract and validate JSON
                json_pattern = r'\{[^{}]*"confidence_score"[^{}]*"message"[^{}]*\}'
                json_match = re.search(json_pattern, response_content, re.DOTALL)
                
                if json_match:
                    try:
                        json_str = json_match.group()
                        json.loads(json_str)  # Validate JSON
                        print(f"✓ Valid JSON found on attempt {attempt + 1}")
                        return {"messages": [HumanMessage(content=response_content)]}
                    except json.JSONDecodeError:
                        pass
            
            # If no valid JSON, try with more explicit prompt
            if attempt < max_attempts - 1:
                print(f"⚠ Attempt {attempt + 1} failed, retrying with more explicit prompt...")
                # Add more explicit instruction for next attempt
                translated.append(HumanMessage(content="Please respond with ONLY a JSON object in this exact format: {\"confidence_score\": 0.XX, \"message\": \"your explanation\"}. No other text."))
            
        except Exception as e:
            print(f"Error in reflection attempt {attempt + 1}: {e}")
            if attempt == max_attempts - 1:
                # Fallback response for final attempt
                fallback_response = '{"confidence_score": 0.5, "message": "Unable to complete automated assessment due to processing error. Manual review recommended."}'
                return {"messages": [HumanMessage(content=fallback_response)]}
    
    # If all attempts failed, return the last response anyway
    return {"messages": [HumanMessage(content=response_content)]}

# Build the graph
builder = StateGraph(State)

# Add all nodes
builder.add_node("generate", generation_node)
builder.add_node("store", external_storage_node, retry=RetryPolicy(max_attempts=3))
builder.add_node("external_process", external_service_node, retry=RetryPolicy(max_attempts=3))
builder.add_node("reflect", reflection_node)
builder.add_node("automatic_approval", external_automation_node)
builder.add_node("human_approval", external_human_node)


# Define the edges
builder.add_edge(START, "generate")
builder.add_edge("generate", "store")
builder.add_edge("store", "external_process")
builder.add_edge("external_process", "reflect")
# Add edges from approval nodes to END
builder.add_edge("automatic_approval", END)
builder.add_edge("human_approval", END)

async def route_after_reflection(state: State) -> str:
    """
    Dynamic router to decide between automatic or human approval
    based on reflection output focusing on hospital verification confidence
    """
    last_message = state["messages"][-1].content
    
    print("=== REFLECTION ROUTING ANALYSIS ===")
    print(f"Raw reflection message: {last_message}")
    print("=" * 50)
    
    try:
        # Clean up the message to extract JSON
        cleaned_message = last_message.strip()
        
        # Handle QwQ model thinking tags - extract content after </think>
        if "<think>" in cleaned_message and "</think>" in cleaned_message:
            # Extract content after the last </think>
            parts = cleaned_message.split("</think>")
            if len(parts) > 1:
                cleaned_message = parts[-1].strip()
        
        # Remove markdown formatting
        cleaned_message = cleaned_message.replace("```json", "").replace("```", "").strip()
        
        # Try multiple JSON extraction approaches
        confidence_score = None
        message_text = ""
        
        # Approach 1: Look for complete JSON object
        import re
        json_pattern = r'\{[^{}]*"confidence_score"[^{}]*"message"[^{}]*\}'
        json_match = re.search(json_pattern, cleaned_message, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group()
                reflection_result = json.loads(json_str)
                confidence_score = reflection_result.get("confidence_score")
                message_text = reflection_result.get("message", "")
            except json.JSONDecodeError:
                pass
        
        # Approach 2: Extract confidence_score value directly
        if confidence_score is None:
            score_pattern = r'"confidence_score":\s*([0-9]*\.?[0-9]+)'
            score_match = re.search(score_pattern, cleaned_message)
            if score_match:
                confidence_score = float(score_match.group(1))
                
                # Try to extract message too
                msg_pattern = r'"message":\s*"([^"]*)"'
                msg_match = re.search(msg_pattern, cleaned_message)
                if msg_match:
                    message_text = msg_match.group(1)
        
        # Approach 3: Look for any decimal number that could be confidence score
        if confidence_score is None:
            # Look for numbers between 0 and 1 that could be confidence scores
            number_pattern = r'\b(0\.[0-9]+|1\.0+|0)\b'
            numbers = re.findall(number_pattern, cleaned_message)
            if numbers:
                # Take the first reasonable confidence score
                for num_str in numbers:
                    num = float(num_str)
                    if 0.0 <= num <= 1.0:
                        confidence_score = num
                        break
        
        if confidence_score is not None:
            print(f"✓ Successfully extracted confidence score: {confidence_score}")
            print(f"✓ Message: {message_text}")
            
            # Hospital verification focused threshold
            requires_human_review = confidence_score < 0.75
            
            if requires_human_review:
                print(f"→ ROUTING TO HUMAN REVIEW (confidence {confidence_score} < 0.75)")
                return "human_approval"
            else:
                print(f"→ ROUTING TO AUTOMATIC APPROVAL (confidence {confidence_score} >= 0.75)")
                return "automatic_approval"
        else:
            print("✗ No confidence score found in reflection")
            print("→ DEFAULTING TO HUMAN REVIEW")
            return "human_approval"
            
    except Exception as e:
        print(f"✗ Unexpected error in routing: {e}")
        print("→ DEFAULTING TO HUMAN REVIEW")
        return "human_approval"

builder.add_conditional_edges(
    "reflect",
    route_after_reflection
)


# Compile the graph with memory checkpointing
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# from langgraph.visualize import visualize
# # Generate the visualization
# viz_graph = visualize(graph)







# Use environment variables for Langfuse configuration
os.environ["LANGFUSE_SECRET_KEY"] = local_secret_key
os.environ["LANGFUSE_HOST"] = langfuse_host
os.environ["LANGFUSE_PUBLIC_KEY"] = local_public_key

from langfuse.langchain import CallbackHandler
 
# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

config = {"configurable": {"thread_id": "1"}, "callbacks": [langfuse_handler]}
topic = "Verify the authenticity of this birth certificate by analyzing the document information and validating the place of birth details."


async def run_agent():
    
    async for event in graph.astream(
        {
            "messages": [
                HumanMessage(content=topic)
            ],
        },
        config,
    ):
        if "generate" in event:
            print("=== BIRTH CERTIFICATE EXTRACTION ===")
            print(event["generate"]["messages"][-1].content)
            print("\n")
        elif "external_process" in event:
            print("=== HOSPITAL VERIFICATION RESULTS ===")
            print(event["external_process"]["messages"][-1].content)
            print("\n")
        elif "reflect" in event:
            print("=== FINAL VERIFICATION ASSESSMENT ===")
            print(event["reflect"]["messages"][-1].content)
            print("\n")
        elif "automatic_approval" in event:
            print("=== ✅ AUTOMATIC APPROVAL ===")
            print("Birth certificate verification passed automated checks")
            print("\n")
        elif "human_approval" in event:
            print("=== 👤 HUMAN REVIEW REQUIRED ===")
            print("Birth certificate requires manual verification")
            print("\n")

import asyncio
asyncio.run(run_agent())
