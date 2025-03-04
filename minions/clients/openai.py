import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai

from minions.usage import Usage


# TODO: define one dataclass for what is returned from all the clients
class OpenAIClient:
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAI client.

        Args:
            model: The name of the model to use (default: "gpt-4o")
            api_key: OpenAI API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
        """
        self.model_name = model
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.logger = logging.getLogger("OpenAIClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                **kwargs,
            }

            # Only add temperature if NOT using the reasoning models (e.g., o3-mini model)
            if "o1" not in self.model_name and "o3" not in self.model_name:
                params["temperature"] = self.temperature

            response = openai.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during OpenAI API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage
