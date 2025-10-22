from langchain_core.output_parsers import JsonOutputParser

from module.LLMs import gpt
from module.prompt import generator_prompt

class DeltaJsonParser:
    def __init__(self):
        self.previous_values = {}
        self.final_values = {} 

    def parse_delta(self, chunk: dict):
        deltas = {}

        for key, value in chunk.items():
            self.final_values[key] = value
            
            if isinstance(value, str):
                prev = self.previous_values.get(key, "")
                if value.startswith(prev):
                    deltas[key] = value[len(prev):]
                    self.previous_values[key] = value
                else:
                    deltas[key] = value
                    self.previous_values[key] = value
            elif isinstance(value, list):
                deltas[key] = value
                
        return deltas
    
    def get_final_values(self):
        return self.final_values.copy()
    
    def reset(self):
        self.previous_values = {}
        self.final_values = {}


json_parser = JsonOutputParser()
delta_parser = DeltaJsonParser()
llm = gpt()

generator_chain = generator_prompt() | llm| json_parser
previous_solution = ""

if __name__ == "__main__":
    query = "독수리 부리는 왜 노랄까?"
    retrieved_bullets = "No related items"

    for chunk in generator_chain.stream({"query": query, "retrieved_bullets": retrieved_bullets}):
        if "solution" in chunk:
            delta = delta_parser.parse_delta(chunk)
            if "solution" in delta:
                print(delta["solution"], end="", flush=True)
    
    final_result = delta_parser.get_final_values()
    print("\n\n")
    print(final_result)
    
    # for chunk in generator_chain.stream({"query" : query, "retrieved_bullets" : retrieved_bullets}):
    #     if "solution" in chunk:
    #         current_solution = chunk['solution']

    #         new_content = current_solution[len(previous_solution):]
    #         print(new_content, end="", flush=True)
    #         previous_solution = current_solution