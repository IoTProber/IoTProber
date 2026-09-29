import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__)))

import json
import anthropic
from google import genai
from openai import OpenAI
from google.genai import types
from util import *
from path_config import LLM_CONFIG_FILE
import logging

def extract_system_and_user_messages(messages):
    """
    检查消息列表中是否有system messages，如果有则提取system content并返回user messages
    
    Args:
        messages: 消息列表，格式如 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    
    Returns:
        tuple: (system_content, user_messages)
            - system_content: system消息的content内容，如果没有则为None
            - user_messages: 不包含system消息的消息列表
    """
    system_content = None
    user_messages = []
    
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content")
        else:
            user_messages.append(msg.get("content"))
    
    return system_content, user_messages

class LLM:
    def __init__(self):
        self.base_path = os.path.dirname(__file__)
        self.llm_config = self.load_llm_config()
        self.llm_initialize()
    
    def load_llm_config(self):
        """加载 LLM 配置文件 """
        if not os.path.exists(LLM_CONFIG_FILE):
            raise FileNotFoundError(f"配置文件不存在: {LLM_CONFIG_FILE}")
        
        with open(LLM_CONFIG_FILE, "r") as f:
            config = json.load(f)
        
        return config

    def llm_initialize(self):                                    
        # self.gemini_client = genai.Client(api_key=self.llm_config["GEMINI"]["API_KEY"])
        self.gemini_client = OpenAI(
            api_key=self.llm_config["GEMINI"]["API_KEY"],
            base_url=self.llm_config["GEMINI"]["BASE_URL"]
        )
        self.deepseek_client = OpenAI(
            api_key=self.llm_config["DEEPSEEK"]["API_KEY"],
            base_url=self.llm_config["DEEPSEEK"]["BASE_URL"]
        )
        self.claude_client = anthropic.Anthropic(
            api_key=self.llm_config["CLAUDE"]["API_KEY"],
            base_url=self.llm_config["CLAUDE"]["BASE_URL"]
        )
        self.openai_client = OpenAI(
            api_key=self.llm_config["OPENAI"]["API_KEY"],
            base_url=self.llm_config["OPENAI"]["BASE_URL"]
        )

    def get_llm_client(self, llm):
        if llm == "DEEPSEEK":
            return self.deepseek_client
        elif llm in ("CLAUDE", "CLAUDEBaseline"):
            return self.claude_client
        elif llm == "GEMINI":
            return self.gemini_client
        elif llm == "OPENAI":
            return self.openai_client
        else:
            return None
    
    def list_models(llm):
        client = get_llm_client(llm)
        models = client.models.list()
        for model in models.data:
            print(f"Model ID: {model.id}")
            print(f"Created at: {model.created_at}")
            print("-" * 20)
        return models
    
    def claude_batch_call(self, llm, batch_messages, whether_json: bool = False):
        """
        params batch_messages: list[list[Dict]], such as [[{"role": "user", "content": "Hello, world"}], [[{"role": "user", "content": "Hi again, friend"}]]]
        """
        client = self.get_llm_client(llm)
        model = self.llm_config[llm]["MODEL"]
        
        requests = []
        for index, message in enumerate(batch_messages):
            single_req = {
                "custom_id": f"req-{index}",
                "params":{
                    "model": model,
                    "max_tokens": 1024,
                    "messages": message
                }
            }
            requests.append(single_req)

        batch = client.messages.batches.create(requests=requests)
        result_stream = client.messages.batches.results(batch.id)

        all_response_content = []
        for entry in result_stream:
            if entry.result.type == "succeeded":
                all_response_content.append(entry.result.message.content)

        return all_response_content

    def chat_with_llm(self, llm, messages, whether_json: bool = False,
                      return_usage: bool = False, max_tokens: int = 1024):
        """
        1. 使用Anthropic SDK调用Claude Sonnet 4.5进行问答
        
        Args:
            user_message: 用户的问题或消息
            api_key: Anthropic API密钥，如果不提供则从环境变量ANTHROPIC_API_KEY读取
        
        Returns:
            Claude的回复内容
        """
        
        client = self.get_llm_client(llm)
        model = self.llm_config[llm]["MODEL"]

        if client is None:
            raise ValueError(f"不支持的 LLM: {llm}")

        if llm == "DEEPSEEK":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                max_tokens=max_tokens,
                # extra_body={"thinking": {"type": "enabled"}},
                response_format={"type": "json_object"} if whether_json else None
            )

            result = json.loads(response.choices[0].message.content)
            if return_usage:
                usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
                return result, usage
            return result
        
        elif llm == "OPENAI":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                # GPT-5-class relays may ignore the legacy max_tokens field
                # and reserve a large default reasoning/output window.
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"} if whether_json else None
            )

            result = response.choices[0].message.content
            if return_usage:
                usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
                return result, usage
            return result

        elif llm in ("CLAUDE", "CLAUDEBaseline"):

            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            
            usage = {"prompt_tokens": response.usage.input_tokens, "completion_tokens": response.usage.output_tokens}
            text_blocks = [b for b in response.content if hasattr(b, 'text')]
            raw_text = text_blocks[0].text if text_blocks else ""
            
            if whether_json:
                try:
                    result = convert_json_from_str(raw_text)
                except Exception:
                    result = {"error": "JSON parse failed", "raw": raw_text[:500]}
            else:
                result = raw_text
            if return_usage:
                return result, usage
            return result
        
        elif llm == "GEMINI":
            # 使用官方API
            # response = client.models.generate_content(
            #     model=model, 
            #     contents=messages
            # )

            # 使用中转站
            response = client.chat.completions.create(
                model=model, 
                messages=messages,
                max_tokens=max_tokens,
            )
            if whether_json:
                result = convert_json_from_str(response.choices[0].message.content)
            else:
                result = response.choices[0].message.content
            if return_usage:
                usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
                return result, usage
            return result
        
        else:
            raise ValueError(f"不支持的 LLM: {llm}")
            
    def batch_chat_with_llm(self, llm, batch_messages):
        if llm != "CLAUDE":
            raise ValueError(f"不支持的 LLM: {llm}")
        
        return self.claude_batch_call(llm, batch_messages)
        
if __name__ == "__main__":
    # 示例1: 单次问答
    llm = LLM()

    print("=== 示例1: 单次问答 ===")
    try:
        user_msg = """
            <user_review>
                The staff at this restaurant are very friendly, and the steak tastes excellent, but the waiting time is too long—I had to wait a full 40 minutes.
            </user_review>

            Please analyze this review and provide the following result in JSON format:
            1. sentiment: positive/negative/neutral
            2. food_quality: string,
            3. service_rating: 1-5,
            4. cons: [con1, con2, ...]
        """

        messages=[
            {"role": "system", "content": f"You are an expert in data analysis and sentiment analysis."},
            {"role": "user", "content": user_msg}
        ]
        
        llm_model = "DEEPSEEK"

        response_json = llm.chat_with_llm(llm_model, messages, whether_json=True)
        print(type(response_json))
        print(f"Deepseek Response: {response_json}\n")
        print(type(response_json["cons"]))

    except Exception as e:
        print(f"错误: {e}\n")
