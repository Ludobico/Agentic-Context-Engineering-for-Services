import tiktoken
import re
from langchain_core.callbacks import AsyncCallbackHandler

def token_calculator(text : str) -> int:
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(text)
    return len(tokens)

class SolutionOnlyStreamCallback(AsyncCallbackHandler):
    def __init__(self):
        self.buffer = ""
        self.in_solution = False
        self.solution_content = ""

    async def on_llm_new_token(self, token : str, **kwargs):
        self.buffer += token
    
        # solution pattern
        if '"solution"' in self.buffer and not self.in_solution:
            match = re.search(r'"solution"\s*:\s*"', self.buffer)
            if match:
                self.in_solution = True
        
        if self.in_solution:
            if token == '"' and self.buffer.count('"solution"') > 0:
                closing_quote_count = self.buffer[self.buffer.rfind('"solution"'):].count('"')
                if closing_quote_count >= 3:
                    self.in_solution = False
                    return
        
            clean_token = token.replace('\\n', '\n').replace('\\"', '"')
            print(clean_token, end="", flush=True)
            self.solution_content += clean_token

    