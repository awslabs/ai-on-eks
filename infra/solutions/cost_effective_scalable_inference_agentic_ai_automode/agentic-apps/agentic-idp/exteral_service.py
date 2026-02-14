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


# External function to verify place of birth information
async def verify_place_of_birth(place_name: str) -> dict:
    """
    Verify if a place of birth is a real location
    """
    try:
        # Known hospitals and medical facilities database (simplified for demo)
        known_hospitals = {
            "armidale and new england hospital": {
                "official_name": "Armidale and New England Hospital",
                "location": "Armidale, New South Wales, Australia",
                "type": "Public Hospital",
                "established": "1950s",
                "status": "Active",
                "verification_sources": [
                    "NSW Health Directory",
                    "Australian Hospital Association",
                    "Google Maps verification"
                ],
                "coordinates": {
                    "latitude": -30.5136,
                    "longitude": 151.6669
                },
                "postal_code": "2350",
                "phone": "+61 2 6776 8888",
                "website": "https://www.health.nsw.gov.au/",
                "services": ["Emergency", "Maternity", "General Medicine", "Surgery"]
            },
            "royal north shore hospital": {
                "official_name": "Royal North Shore Hospital",
                "location": "St Leonards, New South Wales, Australia",
                "type": "Public Hospital",
                "status": "Active"
            },
            "westmead hospital": {
                "official_name": "Westmead Hospital",
                "location": "Westmead, New South Wales, Australia", 
                "type": "Public Hospital",
                "status": "Active"
            }
        }
        
        # Normalize the input for comparison
        normalized_place = place_name.lower().strip()
        
        # Remove common location suffixes that might prevent exact matching
        # Handle cases like "Hospital Name, City" -> "Hospital Name"
        if ", armidale" in normalized_place:
            normalized_place = normalized_place.replace(", armidale", "")
        if ", new south wales" in normalized_place:
            normalized_place = normalized_place.replace(", new south wales", "")
        if ", nsw" in normalized_place:
            normalized_place = normalized_place.replace(", nsw", "")
        if ", australia" in normalized_place:
            normalized_place = normalized_place.replace(", australia", "")
        
        # Check if the place exists in our database (exact match)
        if normalized_place in known_hospitals:
            hospital_info = known_hospitals[normalized_place]
            return {
                "place_verified": True,
                "confidence_score": 0.95,
                "verification_result": {
                    "input_place": place_name,
                    "verified_name": hospital_info["official_name"],
                    "location": hospital_info["location"],
                    "type": hospital_info["type"],
                    "status": hospital_info["status"],
                    "established": hospital_info.get("established", "Unknown"),
                    "coordinates": hospital_info.get("coordinates", {}),
                    "contact_info": {
                        "phone": hospital_info.get("phone", "Not available"),
                        "website": hospital_info.get("website", "Not available")
                    },
                    "services": hospital_info.get("services", []),
                    "verification_sources": hospital_info.get("verification_sources", [])
                },
                "verification_notes": [
                    f"Hospital '{hospital_info['official_name']}' is a verified medical facility",
                    f"Located in {hospital_info['location']}",
                    "Facility is currently active and operational",
                    "Information cross-referenced with official health directories"
                ]
            }
        else:
            # Check for partial matches or similar names
            partial_matches = []
            exact_match_found = False
            
            # First, try to find a hospital name that's contained in the input
            for key, hospital in known_hospitals.items():
                if key in normalized_place or normalized_place in key:
                    # Found a strong partial match - treat as verified
                    return {
                        "place_verified": True,
                        "confidence_score": 0.90,  # Slightly lower than exact match
                        "verification_result": {
                            "input_place": place_name,
                            "verified_name": hospital["official_name"],
                            "location": hospital["location"],
                            "type": hospital["type"],
                            "status": hospital["status"],
                            "established": hospital.get("established", "Unknown"),
                            "coordinates": hospital.get("coordinates", {}),
                            "contact_info": {
                                "phone": hospital.get("phone", "Not available"),
                                "website": hospital.get("website", "Not available")
                            },
                            "services": hospital.get("services", []),
                            "verification_sources": hospital.get("verification_sources", [])
                        },
                        "verification_notes": [
                            f"Hospital '{hospital['official_name']}' is a verified medical facility",
                            f"Located in {hospital['location']}",
                            "Facility is currently active and operational",
                            "Information cross-referenced with official health directories",
                            "Matched via partial name matching (input contained location suffix)"
                        ]
                    }
            
            # If no strong match found, collect weaker partial matches for reporting
            for key, hospital in known_hospitals.items():
                if any(word in normalized_place for word in key.split()) or any(word in key for word in normalized_place.split()):
                    partial_matches.append({
                        "name": hospital["official_name"],
                        "location": hospital["location"]
                    })
            
            return {
                "place_verified": False,
                "confidence_score": 0.2,
                "verification_result": {
                    "input_place": place_name,
                    "status": "Not found in verified database",
                    "partial_matches": partial_matches
                },
                "verification_notes": [
                    f"Place '{place_name}' not found in verified hospital database",
                    "This could indicate a non-existent location or outdated information",
                    "Manual verification recommended for unknown locations"
                ],
                "risk_factors": [
                    "Unverified birth location",
                    "Potential fraudulent document if location doesn't exist",
                    "Requires additional verification through official channels"
                ]
            }
            
    except Exception as e:
        logging.error(f"Place verification error: {str(e)}")
        return {
            "place_verified": False,
            "confidence_score": 0.0,
            "error": str(e),
            "verification_notes": ["Verification service temporarily unavailable"]
        }



# External function to call an API such as google search. it can be hard coded for the moment
async def call_external_service(text: str) -> str:
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
        print(f"Making external Call with data {text}")
        
        # Extract place of birth from the text/messages
        place_of_birth = None
        
        # Parse the text to find place of birth information
        if isinstance(text, list):
            # If text is a list of messages, search through them
            for item in text:
                if hasattr(item, 'content'):
                    content = item.content
                else:
                    content = str(item)
                
                # Look for place_of_birth in multiple formats
                import re
                
                # Method 1: Look for JSON format
                json_match = re.search(r'"place_of_birth":\s*"([^"]+)"', content)
                if json_match:
                    place_of_birth = json_match.group(1)
                    break
                
                # Method 2: Look for text mentioning hospital/place names
                # Look for "Armidale and New England Hospital" or similar patterns
                hospital_patterns = [
                    r'Armidale and New England Hospital[,\s]*Armidale',
                    r'New England Hospital[,\s]*Armidale',
                    r'place of birth[^:]*:\s*"?([^".\n]+(?:Hospital|Medical|Centre)[^".\n]*)"?',
                    r'stated as\s*"([^"]+Hospital[^"]*)"',
                    r'birth[^:]*:\s*"?([^".\n]*Hospital[^".\n]*)"?'
                ]
                
                for pattern in hospital_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        if match.groups():  # Check if there are capture groups
                            place_of_birth = match.group(1).strip()
                        else:
                            place_of_birth = match.group(0).strip()  # Use full match if no groups
                        # Clean up common suffixes
                        place_of_birth = re.sub(r'[,\s]*$', '', place_of_birth)
                        if place_of_birth:
                            break
                
                if place_of_birth:
                    break
        
        # If we found a place of birth, verify it
        if place_of_birth:
            print(f"Verifying place of birth: {place_of_birth}")
            verification_result = await verify_place_of_birth(place_of_birth)
            
            return {
                "verification_type": "place_of_birth",
                "input_data": place_of_birth,
                **verification_result
            }
        else:
            # Debug: Print the content we're trying to parse
            print("DEBUG: Content being parsed:")
            if isinstance(text, list):
                for i, item in enumerate(text):
                    content = item.content if hasattr(item, 'content') else str(item)
                    print(f"Item {i}: {content[:200]}...")
            
            return {
                "verification_type": "place_of_birth",
                "error": "No place of birth information found in the provided data",
                "verification_notes": ["Unable to extract place of birth from document"]
            }
            
    except Exception as e:
        logging.error(f"External service error: {str(e)}")
        return {"error": str(e)}

# Add new node for external processing
async def external_service_node(state: State) -> State:
    """
    Node that calls external processing service
    """
    # Get the last message content
    # last_message = state["messages"][-1].content
    # last_message = state["messages"]
    ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    print(f"AI Messages to Make external call: {json.dumps([msg.content for msg in ai_messages], indent=2)}")
    
    
    # Call external service
    result = await call_external_service(ai_messages)
    
    # Create new message with processed result
    processed_message = HumanMessage(
        content=f"External Processing Results: {json.dumps(result, indent=2)}"
    )
    
    return {"messages": [processed_message]}

