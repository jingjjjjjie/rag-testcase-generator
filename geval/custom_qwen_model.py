# custom llm reference: https://deepeval.com/guides/guides-using-custom-llms
# Add project root to Python path

from src.utils.api_utils import call_api_qwen
from deepeval.models.base_model import DeepEvalBaseLLM

class CustomQwenModel(DeepEvalBaseLLM):
    def __init__(self, model_name="qwen-plus", temperature=0):
        self.model_name = model_name
        self.temperature = temperature

    def load_model(self):
        # No loading needed for API-based model
        return self

    def generate(self, prompt: str) -> str:
        # Call your custom Qwen API
        response, _, _ = call_api_qwen(
            query=prompt,
            model=self.model_name,
            temperature=self.temperature,
            system_prompt=None
        )
        return response

    async def a_generate(self, prompt: str) -> str:
        # For async, just call the sync version
        # (you can make this truly async later if needed)
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name
    
