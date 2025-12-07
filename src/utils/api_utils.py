from dotenv import load_dotenv
from openai import OpenAI
import os
import numpy as np

load_dotenv()

def call_api_qwen(query, model="qwen-plus", temperature=0, system_prompt=None):
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    content = completion.choices[0].message.content
    return content,prompt_tokens,completion_tokens


def get_qwen_embeddings(texts, dim=1024):
    # Refference for embeddings model:
    # https://www.alibabacloud.com/help/en/model-studio/embedding?spm=a2c63.l28256.help-menu-2400256.d_0_8_0.95cf4f2cBwSJ26
    # dim options = 2,048, 1,536, 1,024 (default), 768, 512, 256, 128, 64
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=texts,
        dimensions = dim,
    )

    embeddings = [np.array(data.embedding) for data in response.data]
    total_tokens = response.usage.total_tokens

    return embeddings, total_tokens


def get_qwen_logprobs(prompt, answer):
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,  # Try this - may return null
        top_logprobs=5
    )
    return response

if __name__ == "__main__":
    response, prompt_tokens, completion_tokens = call_api_qwen("Hello, how are you?")
    print("Response:", response)
    print("Prompt Tokens:", prompt_tokens)
    print("Completion Tokens:", completion_tokens)