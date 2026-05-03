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
        config_path = os.path.join(self.base_path, "llm_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, "r") as f:
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
        elif llm == "CLAUDE":
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

    def chat_with_llm(self, llm, messages, whether_json: bool = False):
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
                # extra_body={"thinking": {"type": "enabled"}},
                response_format={"type": "json_object"} if whether_json else None
            )

            return json.loads(response.choices[0].message.content)
        
        elif llm == "OPENAI":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                response_format={"type": "json_object"} if whether_json else None
            )

            return response.choices[0].message.content

        elif llm == "CLAUDE":

            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=1024
            )
            
            if whether_json:
                return convert_json_from_str(response.content[0].text)
            
            return response.content[0].text
        
        elif llm == "GEMINI":
            # 使用官方API
            # response = client.models.generate_content(
            #     model=model, 
            #     contents=messages
            # )

            # 使用中转站
            response = client.chat.completions.create(
                model=model, 
                messages=messages
            )
            if whether_json:
                return convert_json_from_str(response.choices[0].message.content)
            
            return response.choices[0].message.content
        
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