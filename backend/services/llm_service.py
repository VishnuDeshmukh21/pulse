import httpx
import json
from typing import List, Dict, Any

class LLMService:
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url

    async def extract_memory_metadata(self, text: str) -> Dict[str, Any]:
        """Extracts key entity nodes from a raw transcript using Qwen 3B."""
        prompt = f"""
        Analyze this spoken thought and extract metadata in strict JSON format.
        Do not output any introductory or concluding text. Output ONLY valid JSON.
        
        Thought: "{text}"
        
        Desired Format:
        {{
            "entities": ["list", "of", "concrete", "nouns/keywords"],
            "category": "one word general topic classification"
        }}
        """
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }
            )
            res_json = response.json()
            content = res_json["choices"][0]["message"]["content"]
            return json.loads(content)

    async def parse_vague_query(self, query: str) -> Dict[str, Any]:
        """Parses vague historical queries into searchable structured filters."""
        prompt = f"""
        Analyze this query seeking past information. Extract the core search items and the implied time frame.
        Output ONLY valid JSON.
        
        Query: "{query}"
        
        Desired Format:
        {{
            "entities": ["keywords", "to", "look", "for"],
            "days_limit": integers_representing_days_ago_limit
        }}
        
        Example: "a few days back about a bike" -> {{"entities": ["bike"], "days_limit": 7}}
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }
            )
            return json.loads(response.json()["choices"][0]["message"]["content"])

llm_service = LLMService()