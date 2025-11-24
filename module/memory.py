import os
import json
import redis.asyncio as redis

from langchain_core.messages import HumanMessage, AIMessage
from config.getenv import GetEnv

env = GetEnv()

class ResirMemoryManager:
    def __init__(self):
        self.redis_host = env.get_memory_config['REDIS_HOST']
        self.redis_port = env.get_memory_config['REDIS_PORT']
        self.max_memory_size = env.get_memory_config['MAX_MEMORY_SIZE']

        self.r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )

    async def save_user_message(self, session_id : str, user_question : str):
        message = {
            "type" : "user",
            "content" : user_question
        }

        await self.r.lpush(f"session:{session_id}:history", json.dumps(message))

    async def save_ai_message(self, session_id : str, llm_response : str):
        message = {
            "type" : "assistant",
            "content" : llm_response
        }
        await self.r.lpush(f"session:{session_id}:history", json.dumps(message))
    
    async def get_history(self, session_id : str):
        key = f"session:{session_id}:history"
        messages = await self.r.lrange(key, 0, -1)
        return [json.loads(msg) for msg in reversed(messages)]
    
    async def trim_history(self, session_id : str):
        key = f"session:{session_id}:history"
        max_messages = self.max_memory_size
        await self.r.ltrim(key, 0, max_messages - 1)
    
    async def clear_session(self, session_id : str):
        key = f"session:{session_id}:history"
        await self.r.delete(key)

    async def get_langchain_message(self, session_id : str):
        key = f"session:{session_id}:history"
        raw = await self.r.lrange(key, 0, -1)

        parsed = [json.loads(m) for m in reversed(raw)]

        messages = []
        for m in parsed:
            if m['type'] == 'user':
                messages.append(HumanMessage(content=m['content']))
            elif m['type'] == 'assistant':
                messages.append(AIMessage(content=m['content']))
        return messages
    
    