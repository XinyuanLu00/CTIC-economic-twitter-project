import backoff  # For exponential backoff
import openai
import os
import asyncio
from typing import List, Dict, Any

from openai._exceptions import RateLimitError

@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_backoff(**kwargs):
    """Wrapper for OpenAI Completion API with exponential backoff."""
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, RateLimitError)
def chat_completions_with_backoff(**kwargs):
    """Wrapper for OpenAI ChatCompletion API with exponential backoff using new interface."""
    return openai.chat.completions.create(**kwargs)

async def dispatch_openai_chat_requests(
    messages_list: List[List[Dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: List[str]
) -> List[Any]:
    """
    Dispatch requests to OpenAI ChatCompletion API asynchronously
    and return the response objects.
    """
    async_responses = [
        openai.chat.completions.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

async def dispatch_openai_prompt_requests(
    prompt_list: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: List[str]
) -> List[Any]:
    """
    Dispatch prompt-based requests to OpenAI Completion API asynchronously
    and return the response objects.
    """
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop_words
        )
        for x in prompt_list
    ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    """Wrapper for OpenAI API models."""
    def __init__(self, API_KEY: str, model_name: str, stop_words: List[str], max_new_tokens: int):
        openai.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    def chat_generate(self, input_string: str, temperature: float = 0.0) -> str:
        """Generate responses using Chat-based OpenAI models."""
        response = chat_completions_with_backoff(
            model=self.model_name,
            messages=[{"role": "user", "content": input_string}],
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=self.stop_words
        )
        generated_text = response.choices[0].message.content.strip()
        return generated_text

    def prompt_generate(self, input_string: str, temperature: float = 0.0) -> str:
        """Generate responses using prompt-based OpenAI models."""
        response = completions_with_backoff(
            model=self.model_name,
            prompt=input_string,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=self.stop_words
        )
        generated_text = response.choices[0].text.strip()
        return generated_text

    def generate(self, input_string: str, temperature: float = 0.0) -> str:
        """Generate responses using the appropriate model."""
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini']:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized.")

    def batch_chat_generate(self, messages_list: List[str], temperature: float = 0.0) -> List[str]:
        """Batch generate responses using chat-based OpenAI models."""
        open_ai_messages_list = [[{"role": "user", "content": message}] for message in messages_list]
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                open_ai_messages_list,
                self.model_name,
                temperature,
                self.max_new_tokens,
                1.0,
                self.stop_words
            )
        )
        return [res.choices[0].message.content.strip() for res in predictions]

    def batch_prompt_generate(self, prompt_list: List[str], temperature: float = 0.0) -> List[str]:
        """Batch generate responses using prompt-based OpenAI models."""
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                prompt_list,
                self.model_name,
                temperature,
                self.max_new_tokens,
                1.0,
                self.stop_words
            )
        )
        return [res.choices[0].text.strip() for res in predictions]

    def batch_generate(self, messages_list: List[str], temperature: float = 0.0) -> List[str]:
        """Batch generate responses using the appropriate model."""
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini']:
            return self.batch_chat_generate(messages_list, temperature)
        else:
            raise Exception("Model name not recognized.")

    def generate_insertion(self, input_string: str, suffix: str, temperature: float = 0.0) -> str:
        response = completions_with_backoff(
            model=self.model_name,
            prompt=input_string,
            suffix=suffix,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].text.strip()
