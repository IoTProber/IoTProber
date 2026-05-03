from typing import List, Dict, Optional, Any
from enum import Enum
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import warnings

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config import GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY

warnings.filterwarnings("ignore")

class ProblemType(Enum):
    """已知的物联网设备识别问题类型"""
    DEVICE_TYPE = "Identify IoT Device Types"
    DEVICE_VENDOR = "Identify IoT Device Vendors"
    DEVICE_LOCATION = "Identify IoT Device Location"
    DEVICE_MANUFACTURER = "Identify IoT Device Manufacturers"
    DEVICE_MODEL = "Identify IoT Device Models"
    SIMILAR_DEVICES = "Return Similar IoT Devices"

class DecompositionAgent:
    """
    负责用户查询问题的分解，并判断属于哪些已知问题类型
    """
    
    def __init__(self, llm: str = "gemini"):

        self.query_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_db")

        # 已知问题类型列表
        self.known_problems: Dict[str, str] = {
            problem_type.name: problem_type.value 
            for problem_type in ProblemType
        }
        
        # 问题类型关键词映射
        self.problem_keywords = {
            "DEVICE_TYPE": ["type", "kine", "category", "device type", "what device"],
            "DEVICE_MANUFACTURER": ["manufacturer", "vendor", "brand"],
            "DEVICE_MODEL": ["model", "version"],
            "SIMILAR_DEVICES": ["similar", "like", "same"]
        }
        
        # 分解历史记录
        self.decomposition_history: List[Dict[str, Any]] = []

        # 初始化LLM
        self.initialize_llm(llm)
        
    def initialize_llm(self, llm: str = "gemini"):
        """
        初始化LLM
        从config.py导入API密钥
        """
        print("=== 初始化LLM ===")

        os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

        if llm == "gemini":
            print("Loading Gemini 3 pro Model...")
            os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
            load_dotenv()
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-pro-preview",
                temperature=1.0,  # Gemini 3.0+ defaults to 1.0
                max_tokens=None,
                timeout=None
            )
            print("Gemini 3 pro Model loaded.")

        elif llm == "deepseek":
            print("Loading DeepSeek-V3.2 Model...")
            os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
            load_dotenv()
            self.llm = ChatDeepSeek(model="deepseek-chat", temperature=1.3)
            print("DeepSeek Model loaded.")

        elif llm == "openai":
            print("Loading ChatGPT 4 Model...")
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            load_dotenv()
            self.llm = ChatOpenAI(model="gpt-4o", temperature=1.0)
            print("ChatGPT 4 Model loaded.")
        else:
            raise ValueError("Invalid LLM type")
        
        print("=== 初始化LLM完成 === \n")
        
    def decompose_query(self, query: str, rule_matched: bool = False, langchain_version: str = "1.2") -> Dict[str, Any]:
        """
        分解用户查询问题
        
        Args:
            query: 用户输入的查询问题
            
        Returns:
            包含分解结果的字典，格式为:
            {
                "original_query": str,
                "identified_problems": List[str],
                "problem_details": Dict[str, str],
                "is_new_problem": bool,
                "new_problem_description": Optional[str]
            }
        """
        query_lower = query.lower()
        identified_problems = []
        
        if rule_matched:
            # 检查查询是否匹配已知问题类型
            for problem_key, keywords in self.problem_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    identified_problems.append(problem_key)
            
            # 构建问题详情
            problem_details = {
                problem_key: self.known_problems[problem_key]
                for problem_key in identified_problems
            }
            
            # 判断是否为新问题
            is_new_problem = len(identified_problems) == 0
            new_problem_description = None
            
            if is_new_problem:
                # 如果不在已知列表中，添加到问题列表
                new_problem_key = f"CUSTOM_PROBLEM_{len(self.known_problems) + 1}"
                new_problem_description = query
                self.known_problems[new_problem_key] = query
                identified_problems.append(new_problem_key)
                problem_details[new_problem_key] = query
            
            # 记录分解结果
            result = {
                "original_query": query,
                "identified_problems": identified_problems,
                "problem_details": problem_details,
                "is_new_problem": is_new_problem,
                "new_problem_description": new_problem_description
            }
            
            self.decomposition_history.append(result)
            
            return result
        
        else:
            print("Using LLM to decompose query...")

            if langchain_version != "1.2":
                # 使用Langchain Agent v0.3.27 进行问题分解
            
                from langchain.agents import initialize_agent, AgentType
                from langchain.output_parsers import StructuredOutputParser, ResponseSchema
                from langchain.prompts import ChatPromptTemplate

                # Step 1: 构建LLM提示
                prompt = """
                    You are an expert in user needs analysis, capable of accurately breaking down user inquiries.
                    Please analyze the following user query and determine which types of IoT device identification problems it falls under.

                    Known problem types:
                    {known_problem_types}

                    User query: {query}

                    Please return in the following JSON format.
                    {format_instructions}

                    Example 1 - input query: What is the type of this device?
                    Example 1 - output:
                    {{
                        "identified_problems": ["DEVICE_TYPE"],
                        "has_new_problem": false,
                        "new_problem_description": None
                    }}
                    
                    Example 2 - input query: "What is the model and network protocol of this device?"
                    Example 2 - output:
                    {{
                        "identified_problems": ["DEVICE_MODEL", "DEVICE_PROTOCOL"],
                        "has_new_problem": true,
                        "new_problem_description": ["Identify IoT Device Network Protocol"]
                    }}

                    Example 3 - input query: "What is the certificate and the firmware of this device?"
                    Example 3 - output:
                    {{
                        "identified_problems": ["DEVICE_CERTIFICATE", "DEVICE_FIRMWARE"],
                        "has_new_problem": true,
                        "new_problem_description": ["Identify IoT Device Certificates", "Identify IoT Device Firmware"]
                    }}
            """

                # Step 2: 构建输出格式
                ip_schema = ResponseSchema(
                    name="identified_problems",
                    description="List of identified problem type keys. If it is a new problem, provide an new all-caps abbreviation for that problem type."

                )
                op_schema = ResponseSchema(
                    name="has_new_problem",
                    description="Whether has a new problem type (boolean)"
                )
                np_schema = ResponseSchema(
                    name="new_problem_description",
                    description="A list of new problem descriptions. If new problem types appear in the list of identified problem type keys, provide descriptions for these new problems in sequence. \
                        Do not provide descriptions for problem types that already exist. If no new problems, provide None."
                )
                
                response_schema = [ip_schema, op_schema, np_schema]
                
                # Step 3: 构建输出解析器
                output_parser = StructuredOutputParser(response_schemas=response_schema)
                format_instructions = output_parser.get_format_instructions()

                # Step 4: 构建Agent提示词模板和提示词
                agent_prompt = ChatPromptTemplate.from_template(template=prompt)

                known_problem_types = json.dumps(self.known_problems, ensure_ascii=False, indent=2)

                messages = agent_prompt.format_messages(
                    known_problem_types=known_problem_types,
                    query=query,
                    format_instructions=format_instructions
                )

                # Step 5: 构建Agent
                # tools = load_tools(["llm-math", "wikipedia"], llm=self.llm)
                self.agent = initialize_agent(
                    tools=[],
                    llm=self.llm,
                    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    handle_parsing_errors=True, #处理解析错误
                    verbose = False #输出中间步骤
                )

                
                try:
                    # Step 6: 调用Agent
                    response = self.agent.invoke(messages)
                    answer = response["output"]
                    llm_result = output_parser.parse(answer)
                    
                except Exception as e:
                    # Step 9: LLM调用失败时的降级处理
                    print(f"LLM failed! \n Error: {e}")
                    result = {
                        "original_query": query,
                        "identified_problems": [],
                        "problem_details": {},
                        "has_new_problem": True,
                        "new_problem_description": query,
                        "error": str(e)
                    }
                    self.decomposition_history.append(result)
                    return result
            else:
                from pydantic import BaseModel
                from langchain.agents import create_agent
                from langchain.messages import SystemMessage
                from langchain.agents.structured_output import ProviderStrategy

                class DecompositionResult(BaseModel):
                    identified_problems: List[str]
                    has_new_problem: bool
                    new_problem_description: Optional[List[str]]

                    def to_dict(self) -> dict:
                        return {
                            "identified_problems": self.identified_problems,
                            "has_new_problem": self.has_new_problem,
                            "new_problem_description": self.new_problem_description
                        }
                
                # Step 1: 构建Agent提示词
                base_prompt = f"""
                        Please analyze the following user query and determine which types of IoT device identification problems it falls under.

                        Known problem types:
                        {json.dumps(self.known_problems, ensure_ascii=False, indent=2)}

                        User query: {query}

                        Please return in the following JSON format.
                        {{
                            "identified_problems": "List of identified problem type keys. If it is a new problem, provide an new all-caps abbreviation for that problem type.",
                            "has_new_problem": "Whether has a new problem type (boolean).",
                            "new_problem_description": "A list of new problem descriptions. If new problem types appear in the list of identified problem type keys, provide descriptions for these new problems in sequence. 
                                                        Do not provide descriptions for problem types that already exist. If no new problems, provide None."
                        }}
                        
                        Example 1 - input query: What is the type of this device?
                        Example 1 - output:
                        {{
                            "identified_problems": ["DEVICE_TYPE"],
                            "has_new_problem": false,
                            "new_problem_description": None
                        }}
                        
                        Example 2 - input query: "What is the model and network protocol of this device?"
                        Example 2 - output:
                        {{
                            "identified_problems": ["DEVICE_MODEL", "DEVICE_PROTOCOL"],
                            "has_new_problem": true,
                            "new_problem_description": ["Identify IoT Device Network Protocol"]
                        }}

                        Example 3 - input query: "What is the certificate and the firmware of this device?"
                        Example 3 - output:
                        {{
                            "identified_problems": ["DEVICE_CERTIFICATE", "DEVICE_FIRMWARE"],
                            "has_new_problem": true,
                            "new_problem_description": ["Identify IoT Device Certificates", "Identify IoT Device Firmware"]
                        }}

                    """

                question = {"messages": [{"role": "user", "content": base_prompt}]}

                # Step 2: 构建指定输出格式的Agent
                agent = create_agent(
                    model=self.llm,
                    system_prompt=SystemMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "You are an expert in user needs analysis, capable of accurately breaking down user inquiries."
                            }
                        ]
                    ),
                    response_format=ProviderStrategy(DecompositionResult)
                )
                
                # Step 3: 调用Agent
                try:
                    response = agent.invoke(question)
                    llm_result = response["structured_response"].to_dict()
                except Exception as e:
                    print(f"LLM failed! \n Error: {e}")
                    result = {
                        "original_query": query,
                        "identified_problems": [],
                        "problem_details": {},
                        "has_new_problem": True,
                        "new_problem_description": query,
                        "error": str(e)
                    }
                    self.decomposition_history.append(result)
                    return result
                
            # Step 汇总1: 构建问题详情
            problem_details = {
                problem_key: self.known_problems.get(problem_key, "Unknown Problem")
                for problem_key in llm_result["identified_problems"]
            }
            
            # 如果是新问题，添加到已知问题列表
            if llm_result["has_new_problem"] and llm_result["new_problem_description"] is not None:
                new_problem_description_index = 0
                
                for problem_key, problem_description in problem_details.items():
                    if problem_description == "Unknown Problem":
                        problem_details[problem_key] = llm_result["new_problem_description"][new_problem_description_index]
                        new_problem_description_index += 1
                
                        self.known_problems.update({problem_key: problem_details[problem_key]})

                        # 更新ProblemType枚举
                        if not hasattr(ProblemType, problem_key):
                            setattr(ProblemType, problem_key, problem_details[problem_key])

            
            # Step 汇总2: 构建最终结果
            result = {
                "original_query": query,
                "identified_problems": llm_result["identified_problems"],
                "problem_details": problem_details,
                "has_new_problem": llm_result["has_new_problem"],
                "new_problem_description": llm_result["new_problem_description"]
            }
            
            self.decomposition_history.append(result)
            return result

    def add_problem_type(self, problem_key: str, problem_description: str, keywords: Optional[List[str]] = None) -> bool:
        """
        手动添加新的问题类型
        
        Args:
            problem_key: 问题类型的键名
            problem_description: 问题类型的描述
            keywords: 用于识别该问题类型的关键词列表
            
        Returns:
            是否添加成功
        """
        if problem_key in self.known_problems:
            return False
        
        self.known_problems[problem_key] = problem_description
        
        if keywords:
            self.problem_keywords[problem_key] = keywords
        
        return True
    
    def handle_retrieval_response(self, retrieval_result: Optional[Any], original_query: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        处理RetrievalAgent的返回结果
        
        Args:
            retrieval_result: RetrievalAgent返回的结果
            original_query: 原始查询问题
            max_retries: 最大重试次数
            
        Returns:
            处理结果字典，包含:
            {
                "success": bool,
                "retrieval_result": Any,
                "decomposition_result": Dict[str, Any],
                "retry_count": int,
                "message": str
            }
        """
        retry_count = 0
        current_query = original_query
        current_decomposition = None
        
        # 如果retrieval_result为None，尝试重新分解问题
        while retrieval_result is None and retry_count < max_retries:
            retry_count += 1
            
            # 重新分解问题
            current_decomposition = self.decompose_query(current_query)
            
            # 尝试从不同角度重新表述问题
            if retry_count > 1:
                # 如果是第二次及以后的重试，尝试简化或扩展查询
                if len(current_decomposition["identified_problems"]) > 1:
                    # 如果识别出多个问题，尝试只关注第一个
                    first_problem = current_decomposition["identified_problems"][0]
                    current_query = f"{original_query} - 重点关注{self.known_problems[first_problem]}"
                else:
                    # 尝试添加更多上下文
                    current_query = f"{original_query} (物联网设备识别相关)"
            
            # 这里应该调用RetrievalAgent重新检索
            # 由于我们只是处理接口，这里假设外部会重新调用
            break
        
        # 构建返回结果
        result = {
            "success": retrieval_result is not None,
            "retrieval_result": retrieval_result,
            "decomposition_result": current_decomposition or self.decompose_query(original_query),
            "retry_count": retry_count,
            "message": "检索成功" if retrieval_result is not None else f"检索失败，已重试{retry_count}次"
        }
        
        return result
    
    def get_decomposition_history(self) -> List[Dict[str, Any]]:
        """
        获取问题分解历史记录
        
        Returns:
            历史记录列表
        """
        return self.decomposition_history
    
    def get_known_problems(self) -> Dict[str, str]:
        """
        获取当前所有已知问题类型
        
        Returns:
            问题类型字典
        """
        return self.known_problems.copy()
    
    def clear_history(self):
        """清空分解历史记录"""
        self.decomposition_history.clear()

    def save_history(self, filename: str = "decomposition_history.json"):
        with open(os.path.join(self.query_path, filename), "w") as f:
            json.dump(self.decomposition_history, f, indent=4)

    def load_history(self, filename: str = "decomposition_history.json"):
        with open(os.path.join(self.query_path, filename), "r") as f:
            self.decomposition_history = json.load(f)

def main(test_queries):
    # choose_llm = "gemini"
    choose_llm = "deepseek"
    agent = DecompositionAgent(llm=choose_llm)
    
    print("=== 测试问题分解 ===")
    for query in test_queries:
        if choose_llm == "deepseek":
            result = agent.decompose_query(query, rule_matched=False, langchain_version="0.3.27")
        else:
            result = agent.decompose_query(query, rule_matched=False)
        
        print(f"\nQuery: {query}")
        print(f"Identified problems: {result['identified_problems']}")
        print(f"Has new problem: {result['has_new_problem']}")
        if result['has_new_problem']:
            print(f"New problem description: {result['new_problem_description']}\n")

    agent.save_history()
    return result

# 使用示例
if __name__ == "__main__":
    test_queries = [
        "What is the location and whois information of this device?",
        "What is the location and model of this device?"
    ]
    main(test_queries)