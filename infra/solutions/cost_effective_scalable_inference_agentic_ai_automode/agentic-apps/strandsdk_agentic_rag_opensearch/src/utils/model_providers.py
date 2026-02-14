"""Model provider configurations for Strands agents."""

from strands.models.openai import OpenAIModel
from ..config import config

def create_openai_reasoning_model():
    """Create an OpenAI model instance for reasoning tasks."""
    return OpenAIModel(
        client_args={
            "api_key": config.LITELLM_API_KEY,
            "base_url": config.LITELLM_BASE_URL,
        },
        model_id=config.REASONING_MODEL,
        params={
            "temperature": 0.7,
            "max_tokens": 4096,
        }
    )

def get_reasoning_model():
    """Get the configured reasoning model for agents."""
    try:
        # Try to use OpenAI client
        return create_openai_reasoning_model()
    except ImportError:
        # Fallback to string model ID
        return config.REASONING_MODEL
    except Exception as e:
        print(f"Warning: Failed to create OpenAI model, falling back to string ID: {e}")
        return config.REASONING_MODEL
