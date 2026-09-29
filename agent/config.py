"""
agent/config.py — API key shim for decomposition.py / app.py.

The original local config.py (never committed) exported API keys as module
constants. This reconstruction keeps config/llm_config.json as the single source of
truth so keys are never hardcoded here.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_config import LLM_CONFIG_FILE

with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as _f:
    _cfg = json.load(_f)

LLM_CONFIG = _cfg
GEMINI_API_KEY = _cfg["GEMINI"]["API_KEY"]
DEEPSEEK_API_KEY = _cfg["DEEPSEEK"]["API_KEY"]
OPENAI_API_KEY = _cfg["OPENAI"]["API_KEY"]
